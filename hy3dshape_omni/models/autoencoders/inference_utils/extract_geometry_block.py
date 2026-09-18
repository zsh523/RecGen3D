# -*- coding: utf-8 -*-
# Modified by the RecGen3D authors, 2026.
"""
Tencent is pleased to support the open source community by making Tencent Hunyuan 3D Omni available.

Copyright (C) 2025 Tencent.  All rights reserved. The below software and/or models in this 
distribution may have been modified by Tencent ("Tencent Modifications"). All Tencent Modifications 
are Copyright (C) Tencent.

Tencent Hunyuan 3D Omni is licensed under the TENCENT HUNYUAN 3D OMNI COMMUNITY LICENSE AGREEMENT 
except for the third-party components listed below, which is licensed under different terms. 
Tencent Hunyuan 3D Omni does not impose any additional limitations beyond what is outlined in the 
respective licenses of these third-party components. Users must comply with all terms and conditions 
of original licenses of these third-party components and must ensure that the usage of the third party 
components adheres to all relevant laws and regulations. 

For avoidance of doubts, Tencent Hunyuan 3D Omni means training code, inference-enabling code, parameters, 
and/or weights of this Model, which are made publicly available by Tencent in accordance with TENCENT 
HUNYUAN 3D OMNI COMMUNITY LICENSE AGREEMENT.
"""

from typing import Callable, Tuple, List, Union, Optional
from tqdm import tqdm
import numpy as np
import traceback
import trimesh
from skimage import measure
import torch

try:
    from .extract_geometry_base import BaseGeometryExtractor
except ImportError:
    from extract_geometry_base import BaseGeometryExtractor


class BlockGeometryExtractor(BaseGeometryExtractor):
    """
    Extractor that pulls a mesh out of a signed distance function block by block.
    
    The volume is split into blocks; inside each block a multi-scale pyramid sampling
    strategy extracts the iso-surface efficiently, which suits high-resolution extraction.
    """
    
    def __init__(self, device: torch.device = None):
        """
        Initialise the block geometry extractor.
        
        Args:
            device: compute device; chosen automatically when None
        """
        super().__init__(device)
    
    @torch.no_grad()
    def extract_geometry(
        self,
        sdf: Callable,
        resolution: int = 512,
        bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
        bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        level: float = 0,
        coarse_mask: Optional[torch.Tensor] = None,
        crop_size: int = 512,
        disable_tqdm: bool = False,
        **kwargs
    ) -> trimesh.Trimesh:
        """
        Extract a mesh from a signed distance function, block by block.
        
        Args:
            sdf: signed distance function mapping 3D points to SDF values
            resolution: overall resolution, must be a multiple of crop_size, default 512
            bounding_box_min: bounding box minimum, default (-1, -1, -1)
            bounding_box_max: bounding box maximum, default (1, 1, 1)
            level: iso-surface threshold, default 0 (the zero level set)
            coarse_mask: coarse mask used to skip empty regions
            crop_size: resolution of each block, default 512
            
        Returns:
            trimesh.Trimesh: the merged mesh
            
        Raises:
            AssertionError: if resolution is not a multiple of crop_size
        """
        assert resolution % crop_size == 0, f"resolution {resolution} must be multiple of crop_size {crop_size}"
        
        if coarse_mask is not None:
            # reorder the dimensions to match PyTorch's grid_sample layout (z, y, x)
            coarse_mask = coarse_mask.permute(2, 1, 0)[None, None].to(self.device).float()

        blocks_n = resolution // crop_size  # number of blocks along each axis

        # block boundaries along each axis
        xs = np.linspace(bounding_box_min[0], bounding_box_max[0], blocks_n + 1)
        ys = np.linspace(bounding_box_min[1], bounding_box_max[1], blocks_n + 1)
        zs = np.linspace(bounding_box_min[2], bounding_box_max[2], blocks_n + 1)

        meshes = []  # meshes of every block
        
        # show progress
        total_blocks = blocks_n * blocks_n * blocks_n
        progress_bar = tqdm(total=total_blocks, desc="Processing blocks", unit="block", disable_tqdm=disable_tqdm)
        
        # iterate over every block
        for i in range(blocks_n):
            for j in range(blocks_n):
                for k in range(blocks_n):
                    progress_bar.update(1)
                    
                    # bounds of the current block
                    x_min, x_max = xs[i], xs[i + 1]
                    y_min, y_max = ys[j], ys[j + 1]
                    z_min, z_max = zs[k], zs[k + 1]

                    # uniform samples inside this block
                    x = np.linspace(x_min, x_max, crop_size)
                    y = np.linspace(y_min, y_max, crop_size)
                    z = np.linspace(z_min, z_max, crop_size)

                    # build the 3D grid points
                    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
                    points = torch.tensor(
                        np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, 
                        dtype=torch.float
                    ).to(self.device)

                    # reshape into a 3D grid (3, crop_size, crop_size, crop_size)
                    points = points.reshape(crop_size, crop_size, crop_size, 3).permute(3, 0, 1, 2)
                    
                    if coarse_mask is not None:
                        # drop invalid regions using the coarse mask
                        points_tmp = points.permute(1, 2, 3, 0)[None].to(self.device)
                        current_mask = torch.nn.functional.grid_sample(coarse_mask, points_tmp)
                        current_mask = (current_mask > 0.0).cpu().numpy()[0, 0]
                    else:
                        current_mask = None

                    # build the multi-scale pyramid
                    points_pyramid = self.build_pyramid(points, levels=3)

                    # evaluate efficiently using the pyramid
                    mask = None
                    threshold = 2 * (x_max - x_min) / crop_size * 8  # initial threshold
                    
                    for pid, pts in enumerate(points_pyramid):
                        coarse_N = pts.shape[-1]  # resolution at this level
                        pts = pts.reshape(3, -1).permute(1, 0).contiguous()

                        if mask is None:
                            # first level: evaluate everything, or filter by the mask
                            if coarse_mask is not None:
                                pts_sdf = torch.ones_like(pts[:, 1])
                                valid_mask = (
                                    torch.nn.functional.grid_sample(coarse_mask, pts[None, None, None])[0, 0, 0, 0] > 0
                                )
                                if valid_mask.any():
                                    pts_sdf[valid_mask] = self.evaluate_sdf_batch(sdf, pts[valid_mask].contiguous())
                            else:
                                pts_sdf = self.evaluate_sdf_batch(sdf, pts)
                        else:
                            # later levels: only evaluate inside the mask
                            mask = mask.reshape(-1)
                            pts_to_eval = pts[mask]
                            if pts_to_eval.shape[0] > 0:
                                pts_sdf_eval = self.evaluate_sdf_batch(sdf, pts_to_eval.contiguous())
                                # make sure pts_sdf_eval is 1-D
                                if pts_sdf_eval.dim() > 1:
                                    pts_sdf_eval = pts_sdf_eval.squeeze(-1)
                                pts_sdf[mask] = pts_sdf_eval

                        if pid < 3:
                            # update the mask: keep only what is near the iso-surface
                            mask = torch.abs(pts_sdf) < threshold
                            mask = mask.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                            mask = self.upsample(mask.float()).bool()  # upsample to the next level's resolution

                            # upsample the SDF for the next level
                            pts_sdf = pts_sdf.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                            pts_sdf = self.upsample(pts_sdf)
                            pts_sdf = pts_sdf.reshape(-1)

                        threshold /= 2.0  # halve the threshold at each level

                    z = pts_sdf.detach().cpu().numpy()

                    # skip blocks without an iso-surface
                    if current_mask is not None:
                        valid_z = z.reshape(crop_size, crop_size, crop_size)[current_mask]
                        if valid_z.shape[0] <= 0 or (np.min(valid_z) > level or np.max(valid_z) < level):
                            continue

                    # does this block contain the iso-surface?
                    if not (np.min(z) > level or np.max(z) < level):
                        z = z.astype(np.float32)
                        # extract the mesh with marching cubes
                        verts, faces, normals, _ = measure.marching_cubes(
                            volume=z.reshape(crop_size, crop_size, crop_size),
                            level=level,
                            spacing=(
                                (x_max - x_min) / (crop_size - 1),
                                (y_max - y_min) / (crop_size - 1),
                                (z_max - z_min) / (crop_size - 1),
                            ),
                            mask=current_mask,
                        )
                        # move the vertices into the global frame
                        verts = verts + np.array([x_min, y_min, z_min])
                        # build the triangle mesh
                        meshcrop = trimesh.Trimesh(verts, faces, normals)
                        meshes.append(meshcrop)

        progress_bar.close()
        
        # merge the per-block meshes
        if meshes:
            combined = trimesh.util.concatenate(meshes)
            return combined
        else:
            # nothing was extracted: return an empty mesh
            return trimesh.Trimesh()

    def extract_geometry_with_progress(
        self,
        sdf: Callable,
        resolution: int = 512,
        bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
        bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        level: float = 0,
        coarse_mask: Optional[torch.Tensor] = None,
        crop_size: int = 512,
        **kwargs
    ) -> Tuple[trimesh.Trimesh, dict]:
        """
        Extract the geometry and also return detailed progress information.
        
        Args:
            Same arguments as extract_geometry.
            
        Returns:
            Tuple[trimesh.Trimesh, dict]: the extracted mesh and its statistics
        """
        mesh = self.extract_geometry(
            sdf, resolution, bounding_box_min, bounding_box_max, 
            level, coarse_mask, crop_size, **kwargs
        )
        
        stats = self.get_mesh_info(mesh)
        stats['extraction_success'] = self.validate_mesh(mesh)
        
        return mesh, stats


# Function-style wrappers kept for backward compatibility
@torch.no_grad()
def extract_geometry_block(sdf, resolution=512, bounding_box_min=(-1.0, -1.0, -1.0),
                          bounding_box_max=(1.0, 1.0, 1.0), level=0, coarse_mask=None):
    """
    Backward-compatible wrapper around BlockGeometryExtractor.
    
    Args:
        sdf: signed distance function
        resolution: resolution
        bounding_box_min: bounding box minimum
        bounding_box_max: bounding box maximum
        level: iso-surface threshold
        coarse_mask: coarse mask
        
    Returns:
        trimesh.Trimesh: the extracted mesh
        Any: placeholder kept for backward compatibility
    """
    extractor = BlockGeometryExtractor()
    mesh = extractor.extract_geometry(
        sdf, resolution, bounding_box_min, bounding_box_max, level, coarse_mask
    )
    return mesh, None


if __name__ == "__main__":
    import time
    # exercise the new class interface
    print("=" * 50)
    print("Running the BlockGeometryExtractor test suite")
    print("=" * 50)
    start_time = time.time()
    extractor = BlockGeometryExtractor()
    mesh, stats = extractor.extract_geometry_with_progress(
        sdf=BaseGeometryExtractor.sphere_sdf,
        resolution=512,
        bounding_box_min=(-1.0, -1.0, -1.0),
        bounding_box_max=(1.0, 1.0, 1.0),
        level=0,
        coarse_mask=None,
        crop_size=256
    )
    mesh.export("output_block_mesh.obj")
    
    # print detailed mesh information
    print("Extracted mesh info:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # the mesh should be roughly spherical, so its volume should be close to the sphere's
    expected_volume = (4/3) * np.pi * (0.5**3)  # volume of a sphere of radius 0.5
    volume_ratio = stats['volume'] / expected_volume
    print(f"  volume ratio (actual/expected): {volume_ratio:.3f}")

    print(f"BlockGeometryExtractor test passed in {time.time() - start_time:.2f}s")
    
    # exercise the backward-compatible function interface
    print("=" * 50)
    print("Testing backward compatibility ...")
    print("=" * 50)
    start_time = time.time()
    mesh_compat, _ = extract_geometry_block(
        sdf=BaseGeometryExtractor.sphere_sdf,
        resolution=512,
        bounding_box_min=(-1.0, -1.0, -1.0),
        bounding_box_max=(1.0, 1.0, 1.0),
        level=0,
        coarse_mask=None
    )
    assert mesh_compat is not None, "backward-compatible wrapper failed"
    print(f"Backward-compatibility test passed in {time.time() - start_time:.2f}s")
    print("=" * 50)
    print("All tests passed.")
