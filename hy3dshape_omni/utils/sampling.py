# hy3dshape_omni/utils/sampling.py
from typing import Optional, Union, List
import torch
from torch import Tensor

from .procrustes import umeyama_similarity

def fps(
    src: torch.Tensor,
    batch: Optional[Tensor] = None,
    ratio: Optional[Union[Tensor, float]] = None,
    random_start: bool = True,
    batch_size: Optional[int] = None,
    ptr: Optional[Union[Tensor, List[int]]] = None,
):
    from torch_cluster import fps as fps_fn
    return fps_fn(src.float(), batch, ratio, random_start, batch_size, ptr)

def sample_vggt_pcl(tensor_point, tensor_depth, tensor_gt, point_mask, conf, conf_depth, conf_thres=None, target_num_points=4096, uniform_sample=False, return_procrustes=False, return_indices=False, patch_size=14):
    """
    Apply point mask and conf mask
    then use fps or uniform sampling to sample target_num_points points from the tensors.
    
    Args:
        tensor_point: input tensor of shape [B, S, H, W, D]
        tensor_depth: input tensor of shape [B, S, H, W, D] (same shape as tensor_point)
        tensor_gt: input tensor of shape [B, S, H, W, D] (same shape as tensor_point)
        point_mask: boolean mask for points
        conf: confidence values
        conf_depth: confidence values for depth
        conf_thres: confidence threshold percentile (default: 50), or None to skip quantile thresholding
        target_num_points: number of points to sample (default: 4096)
        uniform_sample: if True, use uniform random sampling; if False, use FPS (default: False)
        return_procrustes: if True, apply Procrustes alignment (default: False)
        return_indices: if True, also return the patch-level indices of sampled points (default: False)
        patch_size: size of the patch for computing patch indices (default: 14)
    
    Returns:
        tuple: (result_point, result_conf) or (B_hat, result_conf_depth) if return_procrustes
               If return_indices=True, also returns (sampled_patch_indices, sampled_s_indices) as last element
    """
    # apply confidence threshold
    if conf_thres is not None:
        conf_threshold = torch.quantile(conf.reshape(tensor_point.shape[0], -1), conf_thres / 100.0, dim=1, keepdim=True).view(tensor_point.shape[0], 1,1,1)
        conf_mask = (conf >= conf_threshold) & (conf > 1e-5)
    else:
        # If no threshold specified, only apply minimal confidence check
        conf_mask = conf > 1e-5

    # intersect point mask and confidence mask
    mask = point_mask & conf_mask

    # Reshape: flatten spatial dims but keep batch and feature separate
    B, S, H, W, D = tensor_point.shape
    total_points = S * H * W
    tensor_point_flat = tensor_point.reshape(B, -1, D)
    tensor_depth_flat = tensor_depth.reshape(B, -1, D)
    if tensor_gt is not None:
        tensor_gt_flat = tensor_gt.reshape(B, -1, D)
    conf_flat = conf.reshape(B, -1)  # Flatten conf as well
    conf_depth_flat = conf_depth.reshape(B, -1)  # Flatten conf_depth as well
    mask_flat = mask.reshape(B, -1)
    
    # Create indices tensor to track original positions (pixel-level)
    original_indices = torch.arange(total_points, device=tensor_point.device).unsqueeze(0).expand(B, -1)
    
    # Convert pixel-level indices to patch-level indices
    # Calculate patch dimensions
    patches_per_h = H // patch_size
    patches_per_w = W // patch_size
    patches_per_view = patches_per_h * patches_per_w
    
    # Convert flat indices to (s, h, w) coordinates
    s_coords = original_indices // (H * W)
    h_coords = (original_indices % (H * W)) // W
    w_coords = original_indices % W
    
    # Convert pixel coordinates to patch coordinates
    patch_h = h_coords // patch_size
    patch_w = w_coords // patch_size
    
    # Calculate patch-level flat index
    original_patch_indices = s_coords * patches_per_view + patch_h * patches_per_w + patch_w
    original_s_indices = s_coords  # Store s indices separately
    
    # Calculate max valid count
    num_valid = mask_flat.sum(dim=1)
    max_num = num_valid.max().item()
    
    # Allocate output once for all tensors
    result_point = torch.zeros(B, max_num, D, dtype=tensor_point.dtype, device=tensor_point.device)
    result_depth = torch.zeros(B, max_num, D, dtype=tensor_depth.dtype, device=tensor_depth.device)
    if tensor_gt is not None:
        result_gt = torch.zeros(B, max_num, D, dtype=tensor_gt.dtype, device=tensor_gt.device)
    result_conf = torch.zeros(B, max_num, dtype=conf.dtype, device=conf.device)
    result_conf_depth = torch.zeros(B, max_num, dtype=conf_depth.dtype, device=conf_depth.device)
    result_patch_indices = torch.zeros(B, max_num, dtype=torch.long, device=tensor_point.device)
    result_s_indices = torch.zeros(B, max_num, dtype=torch.long, device=tensor_point.device)
    
    # Use boolean indexing per batch
    for b in range(B):
        n = num_valid[b].item()
        result_point[b, :n] = tensor_point_flat[b][mask_flat[b]]
        result_depth[b, :n] = tensor_depth_flat[b][mask_flat[b]]
        if tensor_gt is not None:
            result_gt[b, :n] = tensor_gt_flat[b][mask_flat[b]]
        result_conf[b, :n] = conf_flat[b][mask_flat[b]]
        result_conf_depth[b, :n] = conf_depth_flat[b][mask_flat[b]]
        result_patch_indices[b, :n] = original_patch_indices[b][mask_flat[b]]
        result_s_indices[b, :n] = original_s_indices[b][mask_flat[b]]

    if uniform_sample:
        # Uniform random sampling - sample only from valid points per batch
        sample_size = target_num_points
        sampled_point = torch.zeros(B, sample_size, D, dtype=result_point.dtype, device=result_point.device)
        sampled_depth = torch.zeros(B, sample_size, D, dtype=result_depth.dtype, device=result_depth.device)
        if tensor_gt is not None:
            sampled_gt = torch.zeros(B, sample_size, D, dtype=result_gt.dtype, device=result_gt.device)
        sampled_conf = torch.zeros(B, sample_size, dtype=result_conf.dtype, device=result_conf.device)
        sampled_conf_depth = torch.zeros(B, sample_size, dtype=result_conf_depth.dtype, device=result_conf_depth.device)
        sampled_patch_indices = torch.zeros(B, sample_size, dtype=torch.long, device=result_point.device)
        sampled_s_indices = torch.zeros(B, sample_size, dtype=torch.long, device=result_point.device)
        
        for b in range(B):
            n_valid = num_valid[b].item()
            if n_valid >= sample_size:
                # No repetition: use randperm to sample without replacement
                indices = torch.randperm(n_valid, device=result_point.device)[:sample_size]
            else:
                # Allow repetition: use randint to sample with replacement
                indices = torch.randint(0, n_valid, (sample_size,), device=result_point.device)
            
            sampled_point[b] = result_point[b, indices]
            sampled_depth[b] = result_depth[b, indices]
            if tensor_gt is not None:
                sampled_gt[b] = result_gt[b, indices]
            sampled_conf[b] = result_conf[b, indices]
            sampled_conf_depth[b] = result_conf_depth[b, indices]
            sampled_patch_indices[b] = result_patch_indices[b, indices]
            sampled_s_indices[b] = result_s_indices[b, indices]
        
        result_point = sampled_point
        result_depth = sampled_depth
        if tensor_gt is not None:
            result_gt = sampled_gt
        result_conf = sampled_conf
        result_conf_depth = sampled_conf_depth
        result_patch_indices = sampled_patch_indices
        result_s_indices = sampled_s_indices
    else:
        # Two-stage sampling: uniform pre-sampling followed by FPS
        pre_sample_size = target_num_points * 4
        
        # Stage 1: Uniform random sampling on valid points per batch
        pre_sampled_point = torch.zeros(B, pre_sample_size, D, dtype=result_point.dtype, device=result_point.device)
        pre_sampled_depth = torch.zeros(B, pre_sample_size, D, dtype=result_depth.dtype, device=result_depth.device)
        if tensor_gt is not None:
            pre_sampled_gt = torch.zeros(B, pre_sample_size, D, dtype=result_gt.dtype, device=result_gt.device)
        pre_sampled_conf = torch.zeros(B, pre_sample_size, dtype=result_conf.dtype, device=result_conf.device)
        pre_sampled_conf_depth = torch.zeros(B, pre_sample_size, dtype=result_conf_depth.dtype, device=result_conf_depth.device)
        pre_sampled_patch_indices = torch.zeros(B, pre_sample_size, dtype=torch.long, device=result_point.device)
        pre_sampled_s_indices = torch.zeros(B, pre_sample_size, dtype=torch.long, device=result_point.device)
        
        for b in range(B):
            n_valid = num_valid[b].item()
            if n_valid >= pre_sample_size:
                # No repetition: use randperm to sample without replacement
                indices = torch.randperm(n_valid, device=result_point.device)[:pre_sample_size]
            else:
                # Allow repetition: use randint to sample with replacement
                indices = torch.randint(0, n_valid, (pre_sample_size,), device=result_point.device)
            
            pre_sampled_point[b] = result_point[b, indices]
            pre_sampled_depth[b] = result_depth[b, indices]
            if tensor_gt is not None:
                pre_sampled_gt[b] = result_gt[b, indices]
            pre_sampled_conf[b] = result_conf[b, indices]
            pre_sampled_conf_depth[b] = result_conf_depth[b, indices]
            pre_sampled_patch_indices[b] = result_patch_indices[b, indices]
            pre_sampled_s_indices[b] = result_s_indices[b, indices]
        
        result_point = pre_sampled_point
        result_depth = pre_sampled_depth
        if tensor_gt is not None:
            result_gt = pre_sampled_gt
        result_conf = pre_sampled_conf
        result_conf_depth = pre_sampled_conf_depth
        result_patch_indices = pre_sampled_patch_indices
        result_s_indices = pre_sampled_s_indices
        
        # Stage 2: FPS sampling on the pre-sampled points (using point coordinates)
        flatten_result_point = result_point.reshape(-1, 3)
        N_down = result_point.shape[1] 
        batch_down = torch.arange(B).to(result_point.device)
        batch_down = torch.repeat_interleave(batch_down, N_down)
        idx_query_random = fps(flatten_result_point, batch_down, ratio=target_num_points / N_down)
        result_point = flatten_result_point[idx_query_random].view(B, -1, D)
        
        # Apply the same FPS indices to depth and conf tensors
        flatten_result_depth = result_depth.reshape(-1, D)
        result_depth = flatten_result_depth[idx_query_random].view(B, -1, D)
        
        if tensor_gt is not None:
            flatten_result_gt = result_gt.reshape(-1, D)
            result_gt = flatten_result_gt[idx_query_random].view(B, -1, D)
        
        flatten_result_conf = result_conf.reshape(-1)
        result_conf = flatten_result_conf[idx_query_random].view(B, -1)
        
        flatten_result_conf_depth = result_conf_depth.reshape(-1)
        result_conf_depth = flatten_result_conf_depth[idx_query_random].view(B, -1)
        
        flatten_result_patch_indices = result_patch_indices.reshape(-1)
        result_patch_indices = flatten_result_patch_indices[idx_query_random].view(B, -1)
        
        flatten_result_s_indices = result_s_indices.reshape(-1)
        result_s_indices = flatten_result_s_indices[idx_query_random].view(B, -1)
    
    if return_procrustes:
        # Apply Procrustes analysis to align the point clouds
        T, s, R, t, B_hat, confidence = umeyama_similarity(result_depth, result_point, weights=result_conf, tensor_point=tensor_point, confidence_check=True, min_confidence=0.85)
        if T is None:
            # Procrustes failed the confidence check; return original points without alignment
            result = (result_point, result_conf, None)
        else:
            result = (B_hat, result_conf_depth, T)
        result += (confidence,)
    else:
        result = (result_point, result_conf, None)
    
    # align gt to the sampled point cloud
    if tensor_gt is not None:
        T_gt, s, R, t, _ = umeyama_similarity(result_gt, result[0], weights=result[1], tensor_point=None)
            #for point in point_map:
            ## Save to obj file with timestamp 
    else:
        T_gt = None
    result = result + (T_gt,)
    
    if return_indices:
        result = result + ((result_patch_indices, result_s_indices),)
    
    return result

def apply_se4_transform_to_surface(surface: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Apply SE(4) transformation (scale + rotation + translation) to surface points.
    
    Args:
        surface: (B, N, 7) tensor with [xyz (3), normal (3), label (1)]
        T: (B, 4, 4) SE(4) transformation matrix
        
    Returns:
        Transformed surface tensor (B, N, 7)
    """
    xyz = surface[:, :, 0:3]      # (B, N, 3)
    normal = surface[:, :, 3:6]   # (B, N, 3)
    label = surface[:, :, 6:]     # (B, N, 1) - keep unchanged
    
    # Transform XYZ: apply full SE(4) transformation using homogeneous coordinates
    ones = torch.ones(*xyz.shape[:-1], 1, device=xyz.device, dtype=xyz.dtype)
    xyz_homo = torch.cat([xyz, ones], dim=-1)  # (B, N, 4)
    xyz_transformed = torch.bmm(xyz_homo, T.transpose(1, 2))[:, :, :3]  # (B, N, 3)
    
    # Transform normals: apply only rotation (upper-left 3x3), no translation
    R = T[:, :3, :3]  # (B, 3, 3) - rotation + scale
    normal_transformed = torch.bmm(normal, R.transpose(1, 2))  # (B, N, 3)
    # Re-normalize since R may include scale
    normal_transformed = normal_transformed / (normal_transformed.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Reconstruct surface tensor
    return torch.cat([xyz_transformed, normal_transformed, label], dim=-1)  # (B, N, 7)
