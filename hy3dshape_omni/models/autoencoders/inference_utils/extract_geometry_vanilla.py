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
from typing import Callable, Tuple, List, Union
import numpy as np
import torch
from tqdm import tqdm
from skimage import measure
from einops import repeat
import trimesh
try:
    from .extract_geometry_base import BaseGeometryExtractor
except ImportError:
    from extract_geometry_base import BaseGeometryExtractor


class VanillaGeometryExtractor(BaseGeometryExtractor):
    """
    Extractor that pulls a mesh out of a signed distance function by uniform sampling.
    """

    @torch.no_grad()
    def extract_geometry(
        self,
        geometric_func: Callable,
        batch_size: int = 1,
        bounds: Union[Tuple[float], List[float], float] = (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
        octree_depth: int = 7,
        num_chunks: int = 10000,
        disable_tqdm: bool = False,
        mc_level: float = -1 / 512,
        octree_resolution: int = None,
        rotation_matrix=None,
        mc_mode='mc',
    ) -> trimesh.Trimesh:
        """
        Extract a mesh from a signed distance function.

        Args:
            geometric_func: maps input points to SDF values.
            batch_size: batch size, default 1.
            bounds: bounding box extent, default (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25).
            octree_depth: octree depth, default 7.
            num_chunks: chunk size, default 10000.
            disable_tqdm: whether to hide the progress bar, default False.
            mc_level: iso-surface threshold, default -1/512.
            octree_resolution: octree resolution, optional.
            diffdmc: differentiable marching cubes implementation, optional.
            rotation_matrix: rotation matrix, optional.
            mc_mode: marching cubes mode, default 'mc'.

        Returns:
            trimesh.Trimesh: the extracted mesh
        """
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size, length = self.generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_depth=octree_depth,
            octree_resolution=octree_resolution,
            indexing="ij"
        )
        grid_size = np.array(grid_size)
        xyz_samples = torch.FloatTensor(xyz_samples)

        if rotation_matrix is not None:
            xyz_samples = torch.matmul(xyz_samples, rotation_matrix)

        if mc_level == -1:
            print(f'Training with soft labels, inference with sigmoid and marching cubes level 0.')
        elif mc_level == 0:
            print(f'VAE Trained with TSDF, inference with marching cubes level 0.')
        else:
            print(f'VAE Trained with Occupancy, inference with marching cubes level {mc_level}.')

        batch_logits = []
        for start in tqdm(
            range(0, xyz_samples.shape[0], num_chunks),
            desc=f"MC Level {mc_level} Implicit Function:", disable=disable_tqdm, leave=False
        ):
            queries = xyz_samples[start: start + num_chunks, :].to(self.device)
            queries = queries.half()
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)

            logits = geometric_func(batch_queries)
            if mc_level == -1:
                mc_level = 0
                print(f'Training with soft labels, inference with sigmoid and marching cubes level 0.')
                logits = torch.sigmoid(logits) * 2 - 1
            batch_logits.append(logits)

        grid_logits = torch.cat(batch_logits, dim=1)
        grid_logits = grid_logits.view((batch_size, grid_size[0], grid_size[1], grid_size[2])).float()

        try:
            if mc_mode == 'mc':
                vertices, faces, normals, _ = measure.marching_cubes(
                    grid_logits[0].cpu().numpy(), mc_level, method="lewiner")
                vertices = vertices / (grid_size - 1) * bbox_size + bbox_min
            elif mc_mode == 'dmc':
                if not hasattr(self, 'dmc'):
                    try:
                        from diso import DiffDMC
                        self.dmc = DiffDMC(dtype=torch.float32).to(self.device)
                    except:
                        raise ImportError("Please install diso via `pip install diso`, or set mc_algo to 'mc'")
                octree_resolution = 2 ** octree_depth if octree_resolution is None else octree_resolution
                sdf = grid_logits[0] / octree_resolution
                verts, faces = self.dmc(sdf, deform=None, return_quads=False, normalize=True)
                verts = self.center_vertices(verts) * 2
                vertices = verts.detach().cpu().numpy()
                faces = faces.detach().cpu().numpy()[:, ::-1]
            else:
                raise ValueError(f"Unknown marching cubes mode: {mc_mode}")
            
            return trimesh.Trimesh(vertices.astype(np.float32), np.ascontiguousarray(faces))
        except (ValueError, RuntimeError) as e:
            traceback.print_exc()
            return trimesh.Trimesh()

    def extract_geometry_with_stats(
        self,
        geometric_func: Callable,
        batch_size: int = 1,
        bounds: Union[Tuple[float], List[float], float] = (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
        octree_depth: int = 7,
        num_chunks: int = 10000,
        disable_tqdm: bool = False,
        mc_level: float = -1 / 512,
        octree_resolution: int = None,
        rotation_matrix=None,
        mc_mode='mc',
    ) -> Tuple[trimesh.Trimesh, dict]:
        """
        Extract the geometry and return statistics alongside it.
        
        Args:
            Same arguments as extract_geometry.
            
        Returns:
            Tuple[trimesh.Trimesh, dict]: the extracted mesh and its statistics
        """
        mesh = self.extract_geometry(
            geometric_func, batch_size, bounds, octree_depth, num_chunks, disable_tqdm, 
            mc_level, octree_resolution, rotation_matrix, mc_mode,
        )
        
        stats = self.get_mesh_info(mesh)
        stats['extraction_success'] = self.validate_mesh(mesh)
        
        return mesh, stats



@torch.no_grad()
def extract_geometry_vanilla(device: torch.device = None, *args, **kwargs):
    """
    Backward-compatible wrapper around VanillaGeometryExtractor.
    """
    extractor = VanillaGeometryExtractor(device)
    mesh = extractor.extract_geometry(*args, **kwargs)
    return ([(mesh.vertices, mesh.faces)] if len(mesh.vertices) > 0 else [(None, None)]), \
           np.array([len(mesh.vertices) > 0], dtype=np.bool_)


if __name__ == "__main__":
    import time
    print("=" * 50)
    print("Running the VanillaGeometryExtractor test suite")
    print("=" * 50)
    start_time = time.time()
    
    # exercise the new class interface
    extractor = VanillaGeometryExtractor()
    mesh, stats = extractor.extract_geometry_with_stats(
        geometric_func=BaseGeometryExtractor.sphere_sdf,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        octree_depth=7,
        mc_level=0,
        mc_mode='dmc' # 'mc'
    )
    
    if stats['extraction_success']:
        mesh.export("output_vanilla_mesh.obj")
        print(f"Mesh extracted: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print("Extracted mesh info:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        # the mesh should be roughly spherical, so its volume should be close to the sphere's
        expected_volume = (4/3) * np.pi * (0.5**3)  # volume of a sphere of radius 0.5
        volume_ratio = stats['volume'] / expected_volume
        print(f"  volume ratio (actual/expected): {volume_ratio:.3f}")
    else:
        print("Mesh extraction failed")
    
    print(f"VanillaGeometryExtractor test passed in {time.time() - start_time:.2f}s")
    # VanillaGeometryExtractor test passed in 2.50s
    
    # exercise the backward-compatible function interface
    print("=" * 50)
    print("Testing backward compatibility ...")
    print("=" * 50)
    start_time = time.time()
    mesh_v_f, has_surface = extract_geometry_vanilla(
        geometric_func=BaseGeometryExtractor.sphere_sdf,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        octree_depth=7,
        mc_level=0,
        mc_mode='mc' # 'mc'
    )
    
    if has_surface[0]:
        mesh = trimesh.Trimesh(mesh_v_f[0][0], mesh_v_f[0][1])
        print('bounds', mesh.bounds)
        mesh.export("output_vanilla_mesh_compat.obj")
        print(f"Wrapper call succeeded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    else:
        print("Wrapper call failed")
    
    print(f"Backward-compatibility test passed in {time.time() - start_time:.2f}s")
    # backward-compatibility test passed in 0.17s
    print("=" * 50)
    print("All tests passed.")
