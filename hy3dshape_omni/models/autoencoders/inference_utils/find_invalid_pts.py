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

from typing import Union, Tuple, List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from tqdm import tqdm


def extract_near_surface_volume_fn(input_tensor: torch.Tensor, alpha: float):
    """
    PyTorch implementation with the dimension handling fixed.
    Args:
        input_tensor: shape [D, D, D], torch.float16
        alpha: scalar offset
    Returns:
        mask: shape [D, D, D], torch.int32 surface mask
    """
    device = input_tensor.device
    D = input_tensor.shape[0]
    signed_val = 0.0

    # apply the offset and deal with invalid values
    val = input_tensor + alpha
    valid_mask = val > -9000  # -9000 marks an invalid value

    # neighbour lookup that keeps the dimensions consistent
    def get_neighbor(t, shift, axis):
        """Shift along the given axis while keeping the dimensions consistent."""
        if shift == 0:
            return t.clone()

        # pick the padding axis (the input [D, D, D] is ordered z, y, x)
        pad_dims = [0, 0, 0, 0, 0, 0]  # order: [x before, x after, y before, y after, z before, z after]

        # set the padding for this axis
        if axis == 0:  # x axis (last dimension)
            pad_idx = 0 if shift > 0 else 1
            pad_dims[pad_idx] = abs(shift)
        elif axis == 1:  # y axis (middle dimension)
            pad_idx = 2 if shift > 0 else 3
            pad_dims[pad_idx] = abs(shift)
        elif axis == 2:  # z axis (first dimension)
            pad_idx = 4 if shift > 0 else 5
            pad_dims[pad_idx] = abs(shift)

        # pad, adding batch and channel dimensions for F.pad
        padded = F.pad(t.unsqueeze(0).unsqueeze(0), pad_dims[::-1], mode='replicate')  # reversed to match F.pad's ordering

        # build the slice index
        slice_dims = [slice(None)] * 3  # start with a full slice
        if axis == 0:  # x axis (dim=2)
            if shift > 0:
                slice_dims[0] = slice(shift, None)
            else:
                slice_dims[0] = slice(None, shift)
        elif axis == 1:  # y axis (dim=1)
            if shift > 0:
                slice_dims[1] = slice(shift, None)
            else:
                slice_dims[1] = slice(None, shift)
        elif axis == 2:  # z axis (dim=0)
            if shift > 0:
                slice_dims[2] = slice(shift, None)
            else:
                slice_dims[2] = slice(None, shift)

        # apply the slice and restore the shape
        padded = padded.squeeze(0).squeeze(0)
        sliced = padded[slice_dims]
        return sliced

    # neighbours along each direction (shapes stay consistent)
    left = get_neighbor(val, 1, axis=0)  # x direction
    right = get_neighbor(val, -1, axis=0)
    back = get_neighbor(val, 1, axis=1)  # y direction
    front = get_neighbor(val, -1, axis=1)
    down = get_neighbor(val, 1, axis=2)  # z direction
    up = get_neighbor(val, -1, axis=2)

    # handle invalid values at the border (where keeps the shapes consistent)
    def safe_where(neighbor):
        return torch.where(neighbor > -9000, neighbor, val)

    left = safe_where(left)
    right = safe_where(right)
    back = safe_where(back)
    front = safe_where(front)
    down = safe_where(down)
    up = safe_where(up)

    # check sign agreement (in float32 for precision)
    sign = torch.sign(val.to(torch.float32))
    neighbors_sign = torch.stack([
        torch.sign(left.to(torch.float32)),
        torch.sign(right.to(torch.float32)),
        torch.sign(back.to(torch.float32)),
        torch.sign(front.to(torch.float32)),
        torch.sign(down.to(torch.float32)),
        torch.sign(up.to(torch.float32))
    ], dim=0)

    # are all the signs the same?
    same_sign = torch.all(neighbors_sign == sign, dim=0)

    # build the final mask
    mask = (~same_sign).to(torch.int32)
    return mask * valid_mask.to(torch.int32)
