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

import numpy as np
import torch
from typing import Callable, Tuple, List, Union, Optional
import trimesh
from abc import ABC, abstractmethod


class BaseGeometryExtractor(ABC):
    """Base class for geometry extractors, holding the shared processing helpers."""
    
    def __init__(self, device: torch.device = None):
        """
        Initialise the geometry extractor.
        
        Args:
            device: compute device; chosen automatically when None
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_pooling_operations()
    
    def _setup_pooling_operations(self):
        """Set up the 3D pooling and upsampling operations."""
        self.avg_pool_3d = torch.nn.AvgPool3d(2, stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
        self.max_pool_3d = torch.nn.MaxPool3d(3, stride=1, padding=1)
    
    def generate_dense_grid_points(
            self,
            bbox_min: np.ndarray,
            bbox_max: np.ndarray,
            octree_depth: int = 7,
            octree_resolution: int = None,
            indexing: str = "ij"
        ) -> Tuple[np.ndarray, List[int], np.ndarray]:
        '''
        Generate dense grid point coordinates from the bounding box and octree depth.
        
        Args:
            - bbox_min: bounding box minimum (3D coordinate)
            - bbox_max: bounding box maximum (3D coordinate)
            - octree_depth: octree depth, default 7
            - indexing: grid indexing convention, default "ij"
            - octree_resolution: octree resolution, optional
            
        Returns:
            - xyz: the generated grid point coordinates
            - grid_size: grid dimensions
            - length: bounding box side length
        '''
        length = bbox_max - bbox_min
        num_cells = octree_resolution or np.exp2(octree_depth)

        # build the linearly spaced coordinates
        x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
        y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
        z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
        [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
        xyz = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)
        grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

        return xyz, grid_size, length


    def evaluate_sdf_batch(
        self, 
        sdf_func: Callable, 
        points: torch.Tensor, 
        batch_size: int = 100000
    ) -> torch.Tensor:
        """
        Evaluate the SDF in batches to avoid running out of memory.
        
        Args:
            sdf_func: the SDF evaluation function
            points: input points
            batch_size: batch size
            
        Returns:
            the SDF value of every point
        """
        z = []
        for _, pnts in enumerate(torch.split(points, batch_size, dim=0)):
            value = sdf_func(pnts.unsqueeze(0)).squeeze(0)
            z.append(value)
        return torch.cat(z, axis=0)
    
    def build_pyramid(self, points: torch.Tensor, levels: int = 3) -> List[torch.Tensor]:
        """
        Build a multi-scale pyramid.
        
        Args:
            points: input points, shape [3, H, W, D]
            levels: number of pyramid levels
            
        Returns:
            the pyramid, from fine to coarse
        """
        pyramid = [points]
        for _ in range(levels):
            points = self.avg_pool_3d(points[None])[0]
            pyramid.append(points)
        return pyramid[::-1]  # reverse the order: coarse to fine

    @staticmethod
    def find_skippable_cube(grid):
        empty_counter = 0
        occupy_counter = 0
        for i in range(0, grid.shape[0] - 1):
            for j in range(0, grid.shape[0] - 1):
                for k in range(0, grid.shape[0] - 1):
                    v1 = grid[i, j, k]
                    v2 = grid[i + 1, j, k]
                    v3 = grid[i, j + 1, k]
                    v4 = grid[i, j, k + 1]
                    v5 = grid[i + 1, j + 1, k]
                    v6 = grid[i + 1, j, k + 1]
                    v7 = grid[i, j + 1, k + 1]
                    v8 = grid[i + 1, j + 1, k + 1]
                    if np.sign(v1) == np.sign(v2) == np.sign(v3) == np.sign(v4) == \
                        np.sign(v5) == np.sign(v6) == np.sign(v7) == np.sign(v8):
                        # The signs are all the same
                        empty_counter += 1
                    else:
                        # The signs are not all the same
                        occupy_counter += 1

        return empty_counter, occupy_counter

    @staticmethod
    def center_vertices(vertices):
        """Translate the vertices so that bounding box is centered at zero."""
        vert_min = vertices.min(dim=0)[0]
        vert_max = vertices.max(dim=0)[0]
        vert_center = 0.5 * (vert_min + vert_max)
        return vertices - vert_center
        
    @abstractmethod
    def extract_geometry(self, sdf_func: Callable, **kwargs):
        """
        Abstract method: extract the geometry.
        
        Args:
            sdf_func: the SDF function
            **kwargs: further arguments
            
        Returns:
            the extracted mesh
        """
        pass
    
    def validate_mesh(self, mesh: trimesh.Trimesh) -> bool:
        """
        Check the basic properties of a mesh.
        
        Args:
            mesh: the mesh to check
            
        Returns:
            whether the mesh passed
        """
        if mesh is None:
            return False
        if len(mesh.vertices) == 0:
            return False
        if len(mesh.faces) == 0:
            return False
        return True
    
    def get_mesh_info(self, mesh: trimesh.Trimesh) -> dict:
        """
        Collect information about a mesh.
        
        Args:
            mesh: the mesh object
            
        Returns:
            a dict of mesh statistics
        """
        return {
            'num_vertices': len(mesh.vertices),
            'num_faces': len(mesh.faces),
            'volume': mesh.volume,
            'bounds': mesh.bounds,
            'is_watertight': mesh.is_watertight
        }

    @staticmethod
    def sphere_sdf(points: torch.Tensor, radius: float = 0.5) -> torch.Tensor:
        """
        Signed distance function of a sphere.
        
        Args:
            points: input points, shape [batch_size, num_points, 3] or [num_points, 3]
            radius: sphere radius, default 0.5
            
        Returns:
            SDF values, same shape as the input points minus the last dimension
        """
        if points.dim() == 3:
            distances = torch.norm(points, dim=-1, keepdim=True)
        else:
            distances = torch.norm(points, dim=-1, keepdim=True)
        return distances - radius

    @staticmethod
    def cube_sdf(points: torch.Tensor, size_length: float = 1.0) -> torch.Tensor:
        """
        Signed distance function of a cube.
        
        Args:
            points: input points
            
        Returns:
            SDF values
        """
        half_size = size_length / 2
        return torch.max(torch.abs(points), dim=-1, keepdim=True)[0] - half_size
