# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Modified by the RecGen3D authors, 2026.

import torch
import logging
from typing import Optional, Tuple
from vggt.utils.geometry import closed_form_inverse_se3

def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.
    
    Args:
        input_tensor (torch.Tensor): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Values outside 
                                  [-hard_max, hard_max] will be clamped. If None, 
                                  no clamping is performed. Defaults to 100.
    """
    if input_tensor is None:
        return input_tensor
    
    # Check for inf/nan values
    has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
    if has_inf_nan:
        logging.warning(f"Tensor {loss_name} contains inf or nan values. Replacing with zeros.")
        input_tensor = torch.where(
            torch.isnan(input_tensor) | torch.isinf(input_tensor),
            torch.zeros_like(input_tensor),
            input_tensor
        )

    # Apply hard clamping if specified
    if hard_max is not None:
        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

    return input_tensor


def check_valid_tensor(input_tensor: Optional[torch.Tensor], name: str = "tensor") -> None:
    """
    Check if a tensor contains NaN or Inf values and log a warning if found.
    
    Args:
        input_tensor: The tensor to check
        name: Name of the tensor for logging purposes
    """
    if input_tensor is not None:
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            logging.warning(f"NaN or Inf found in tensor: {name}")


def normalize_camera_extrinsics_and_points_batch(
    extrinsics: torch.Tensor,
    cam_points: Optional[torch.Tensor] = None,
    world_points: Optional[torch.Tensor] = None,
    depths: Optional[torch.Tensor] = None,
    scale_by_points: bool = True,
    point_masks: Optional[torch.Tensor] = None,
    HY_coord = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Normalize camera extrinsics and corresponding 3D points.
    
    This function scales the scene to have unit average distance.
    
    Args:
        extrinsics: Camera extrinsic matrices of shape (B, S, 3, 4)
        cam_points: 3D points in camera coordinates of shape (B, S, H, W, 3) or (*,3)
        world_points: 3D points in world coordinates of shape (B, S, H, W, 3) or (*,3)
        depths: Depth maps of shape (B, S, H, W)
        scale_by_points: Whether to normalize the scale based on point distances
        point_masks: Boolean masks for valid points of shape (B, S, H, W)
    
    Returns:
        Tuple containing:
        - Normalized camera extrinsics of shape (B, S, 3, 4)
        - Normalized camera points (same shape as input cam_points)
        - Normalized world points (same shape as input world_points)
        - Normalized depths (same shape as input depths)
    """
    # Validate inputs
    check_valid_tensor(extrinsics, "extrinsics")
    check_valid_tensor(cam_points, "cam_points")
    check_valid_tensor(world_points, "world_points")
    check_valid_tensor(depths, "depths")


    B, S, _, _ = extrinsics.shape
    device = extrinsics.device


    if HY_coord:
        extrinsics_homog = extrinsics
        new_extrinsics = extrinsics_homog.detach().clone()
    else:
        # Convert extrinsics to homogeneous form: (B, N,4,4)
        #extrinsics_homog = torch.cat(
            #[
                #extrinsics,
                #torch.zeros((B, S, 1, 4), device=device),
            #],
        #)
        extrinsics_homog = extrinsics

        # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
        # which can be also viewed as the cam_to_world extrinsic matrix
        first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
        # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
        new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)


    if world_points is not None:
        # since we are transforming the world points to the first camera's coordinate system
        # we directly use the cam_from_world extrinsic matrix of the first camera
        # instead of using the inverse of the first camera's extrinsic matrix
        R = extrinsics[:, 0, :3, :3]
        t = extrinsics[:, 0, :3, 3]
        if HY_coord:
            temp_world_points = (world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            new_world_points = world_points
        else:
            new_world_points = (world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        new_world_points = None


    if scale_by_points:
        new_cam_points = cam_points.clone()
        new_depths = depths.clone()

        if HY_coord:
            dist = temp_world_points.norm(dim=-1)
        else:
            dist = new_world_points.norm(dim=-1)
        dist_sum = (dist * point_masks).sum(dim=[1,2,3])
        valid_count = point_masks.sum(dim=[1,2,3])
        avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)


        new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        if depths is not None:
            new_depths = new_depths / avg_scale.view(-1, 1, 1, 1)
        if cam_points is not None:
            new_cam_points = new_cam_points / avg_scale.view(-1, 1, 1, 1, 1)
    else:
        return new_extrinsics[:, :, :3], cam_points, new_world_points, depths

    new_extrinsics = check_and_fix_inf_nan(new_extrinsics, "new_extrinsics", hard_max=None)
    new_cam_points = check_and_fix_inf_nan(new_cam_points, "new_cam_points", hard_max=None)
    new_world_points = check_and_fix_inf_nan(new_world_points, "new_world_points", hard_max=None)
    new_depths = check_and_fix_inf_nan(new_depths, "new_depths", hard_max=None)


    if HY_coord:
        avg_scale = avg_scale.log().unsqueeze(1).expand(-1, new_extrinsics.shape[1])
        return new_extrinsics, new_cam_points, new_world_points, new_depths, avg_scale
    else:
        return new_extrinsics, new_cam_points, new_world_points, new_depths


def normalize_camera_extrinsics_and_points_batch_wrapper(batch):
    # Normalize camera extrinsics and points. The function returns new tensors.
    normalized_extrinsics_HY, normalized_cam_points_HY, normalized_world_points_HY, normalized_depths_HY, scale_HY = \
        normalize_camera_extrinsics_and_points_batch(
            extrinsics=batch["extrinsics"],
            cam_points=batch["cam_points"],
            world_points=batch["world_points"],
            depths=batch["depths"],
            point_masks=batch["point_masks"],
            HY_coord= True,
        )

    normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = \
        normalize_camera_extrinsics_and_points_batch(
            extrinsics=batch["extrinsics"],
            cam_points=batch["cam_points"],
            world_points=batch["world_points"],
            depths=batch["depths"],
            point_masks=batch["point_masks"],
            HY_coord= False,
        )

    # for render visualization
    batch["extrinsics_render"] = batch["extrinsics"]

    # Replace the original values in the batch with the normalized ones.
    batch["extrinsics"] = normalized_extrinsics
    batch["cam_points"] = normalized_cam_points
    batch["world_points"] = normalized_world_points
    batch["depths"] = normalized_depths

    # set extrinsics for HY coordinate system
    batch["extrinsics_HY"] = normalized_extrinsics_HY
    batch["scale_HY"] = scale_HY

    batch["world_points"] = normalized_world_points_HY * scale_HY.exp().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # set point for HY coordinate system
    indices = torch.linspace(0, batch['surface'].shape[1] - 1, 4096).long()  # shape (4096,)
    batch["point"] = batch['surface'][:,indices,:][:,:,:3]

    # for safety
    batch["point"] *= 0 

    return batch


