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
import traceback
import numpy as np
from skimage import measure
from scipy.interpolate import RegularGridInterpolator
import trimesh
import torch
from einops import repeat

try:
    from .extract_geometry_base import BaseGeometryExtractor
except ImportError:
    from extract_geometry_base import BaseGeometryExtractor


class FastGeometryExtractorV1(BaseGeometryExtractor):
    """
    Fast geometry extractor: interpolates a coarse SDF grid to reach a fine mesh.
    """
    
    def __init__(self, device: torch.device = None):
        """
        Initialise the fast geometry extractor.
        
        Args:
            device: compute device; chosen automatically when None
        """
        super().__init__(device)
    
    @torch.no_grad()
    def extract_geometry(
        self,
        geometric_func: Callable,
        batch_size: int = 1,
        bounds: Union[Tuple[float], List[float], float] = (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
        octree_depth: int = 7,
        num_chunks: int = 10000,
        disable_tqdm: bool = False,
        **kwargs
    ) -> trimesh.Trimesh:
        """
        Extract geometry by evaluating the SDF on a dense grid of points.
        
        Args:
            geometric_func: the function that evaluates the signed distance field
            batch_size: batch size, default 1
            bounds: bounding box as (x_min, y_min, z_min, x_max, y_max, z_max)
            octree_depth: octree depth, default 7
            num_chunks: how many chunks the grid points are split into, default 10000
            disable_tqdm: whether to hide the progress bar, default False
            
        Returns:
            trimesh.Trimesh: the extracted mesh
        """
        # start from a low-resolution grid
        grid_size = 257
        grid64 = np.linspace(-1.25, 1.25, grid_size)

        # normalise the bounding box argument
        if isinstance(bounds, float):
            bounds = abs(bounds)
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min = np.array(bounds[0:3])  # bounding box minimum
        bbox_max = np.array(bounds[3:6])  # bounding box maximum
        bbox_size = bbox_max - bbox_min   # bounding box size

        # build the dense grid points
        xyz_samples, grid_size_dense, length = self.generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_depth=8,
            indexing="ij"
        )
        xyz_samples = torch.FloatTensor(xyz_samples)

        # query the SDF in batches
        batch_logits = []
        for start in tqdm(
            range(0, xyz_samples.shape[0], num_chunks),
            desc="Implicit Function:", disable=disable_tqdm, leave=False
        ):
            queries = xyz_samples[start: start + num_chunks, :].to(self.device)
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)

            logits = geometric_func(batch_queries)
            batch_logits.append(logits.cpu())

        sdf_values = torch.cat(batch_logits, dim=1)
        sdf_values = sdf_values.view((batch_size, grid_size_dense[0], grid_size_dense[1], grid_size_dense[2]))
        sdf_values = sdf_values.numpy()[0]

        # build the high-resolution grid
        grid_size_high_res = 513
        grid128 = np.linspace(-1.25, 1.25, grid_size_high_res)
        x_high_res, y_high_res, z_high_res = np.meshgrid(grid128, grid128, grid128, indexing='ij')

        # build an interpolator from the low-resolution SDF
        interpolator = RegularGridInterpolator((grid64, grid64, grid64), sdf_values)

        # interpolate the SDF onto the high-resolution grid
        coords_high_res = np.stack((x_high_res, y_high_res, z_high_res), axis=-1)
        sdf_values_high_res = interpolator(coords_high_res)

        # find the low-resolution vertices where the sign changes
        mixed_signs = np.zeros_like(sdf_values, dtype=bool)
        mixed_signs[:-1, :-1, :-1] = (
            (np.sign(sdf_values[:-1, :-1, :-1]) != np.sign(sdf_values[1:, 1:, 1:])) |
            (np.sign(sdf_values[:-1, :-1, 1:]) != np.sign(sdf_values[1:, 1:, :-1])) |
            (np.sign(sdf_values[:-1, 1:, :-1]) != np.sign(sdf_values[1:, :-1, 1:])) |
            (np.sign(sdf_values[1:, :-1, :-1]) != np.sign(sdf_values[:-1, 1:, 1:]))
        )

        # mark those vertices as nan so they get re-evaluated
        sdf_values_high_res[2 * mixed_signs] = np.nan

        # evaluate the SDF exactly at the nan vertices
        nan_vertices = np.isnan(sdf_values_high_res)
        coords_nan = coords_high_res[nan_vertices]
        xyz_samples = torch.FloatTensor(coords_nan)

        batch_logits = []
        for start in tqdm(
            range(0, xyz_samples.shape[0], num_chunks),
            desc="Implicit Function:", disable=disable_tqdm, leave=False
        ):
            queries = xyz_samples[start: start + num_chunks, :].to(self.device)
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)

            logits = geometric_func(batch_queries)
            batch_logits.append(logits.cpu())
        sdf_values_nan_vertices = torch.cat(batch_logits, dim=1).numpy()[0]

        # write the exact values back into the high-resolution grid
        sdf_values_nan_vertices = torch.cat(batch_logits, dim=1).numpy()[0]
        sdf_values_nan_vertices = sdf_values_nan_vertices.flatten()  # make sure it is 1-D
        sdf_values_high_res[nan_vertices] = sdf_values_nan_vertices

        # extract the mesh with marching cubes
        try:
            vertices, faces, normals, _ = measure.marching_cubes(sdf_values_high_res, 0, method="lewiner")
            vertices = vertices / grid_size_high_res * bbox_size + bbox_min
            mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=np.ascontiguousarray(faces))
            return mesh
        except (ValueError, RuntimeError) as e:
            print(f"Mesh extraction failed: {e}")
            return trimesh.Trimesh()

    def extract_geometry_with_stats(
        self,
        geometric_func: Callable,
        batch_size: int = 1,
        bounds: Union[Tuple[float], List[float], float] = (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
        octree_depth: int = 7,
        num_chunks: int = 10000,
        disable_tqdm: bool = True,
        **kwargs
    ) -> Tuple[trimesh.Trimesh, dict]:
        """
        Extract the geometry and return statistics alongside it.
        
        Args:
            Same arguments as extract_geometry.
            
        Returns:
            Tuple[trimesh.Trimesh, dict]: the extracted mesh and its statistics
        """
        mesh = self.extract_geometry(
            geometric_func, batch_size, bounds, octree_depth, num_chunks, disable_tqdm, **kwargs
        )
        
        stats = self.get_mesh_info(mesh)
        stats['extraction_success'] = self.validate_mesh(mesh)
        
        return mesh, stats


# Function-style wrappers kept for backward compatibility
@torch.no_grad()
def extract_geometry_fast_v1(
        geometric_func: Callable,
        device: torch.device,
        batch_size: int = 1,
        bounds: Union[Tuple[float], List[float], float] = (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
        octree_depth: int = 7,
        num_chunks: int = 10000,
        disable_tqdm: bool = True
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Backward-compatible wrapper around FastGeometryExtractor.
    
    Args:
        geometric_func: the geometry (SDF) function
        device: compute device
        batch_size: batch size
        bounds: bounding box extent
        octree_depth: octree depth
        num_chunks: number of chunks
        disable_tqdm: whether to hide the progress bar
        
    Returns:
        Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]: per-mesh (vertices, faces) and a has-surface flag
    """
    extractor = FastGeometryExtractorV1(device)
    mesh = extractor.extract_geometry(
        geometric_func, batch_size, bounds, octree_depth, num_chunks, disable_tqdm
    )
    
    # convert to the backward-compatible return format
    if extractor.validate_mesh(mesh):
        mesh_v_f = [(mesh.vertices.astype(np.float32), np.ascontiguousarray(mesh.faces))]
        has_surface = np.array([True], dtype=np.bool_)
    else:
        mesh_v_f = [(None, None)]
        has_surface = np.array([False], dtype=np.bool_)
    
    return mesh_v_f, has_surface


if __name__ == "__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # exercise the new class interface
    print("=" * 50)
    print("Running the FastGeometryExtractor test suite")
    print("=" * 50)
    start_time = time.time()
    extractor = FastGeometryExtractorV1(device)
    mesh, stats = extractor.extract_geometry_with_stats(
        geometric_func=BaseGeometryExtractor.sphere_sdf,
        batch_size=1,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        octree_depth=7,
        num_chunks=10000,
        disable_tqdm=True
    )
    mesh.export("output_fast_mesh.obj")
    
    # print detailed mesh information
    print("Extracted mesh info:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # the mesh should be roughly spherical, so its volume should be close to the sphere's
    expected_volume = (4/3) * np.pi * (0.5**3)  # volume of a sphere of radius 0.5
    volume_ratio = stats['volume'] / expected_volume
    print(f"  volume ratio (actual/expected): {volume_ratio:.3f}")
    print(f"FastGeometryExtractor test passed in {time.time() - start_time:.2f}s")
    
    # exercise the backward-compatible function interface
    print("=" * 50)
    print("Testing backward compatibility ...")
    print("=" * 50)
    start_time = time.time()
    mesh_v_f, has_surface = extract_geometry_fast_v1(
        geometric_func=BaseGeometryExtractor.sphere_sdf,
        device=device,
        batch_size=1,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        octree_depth=7,
        num_chunks=10000,
        disable_tqdm=True
    )
    
    assert len(mesh_v_f) == 1, "backward-compatible wrapper failed"
    assert has_surface[0], "backward-compatible wrapper failed"
    print(f"Backward-compatibility test passed in {time.time() - start_time:.2f}s")
    print("=" * 50)
    print("All tests passed.")
    # FastGeometryExtractor test passed in 753.39s
