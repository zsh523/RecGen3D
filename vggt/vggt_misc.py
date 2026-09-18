import os
import json
import torch
from collections import deque
from torch import inf
from accelerate import Accelerator
import trimesh
import numpy as np
import torch
from vggt.visual_util import predictions_to_glb_w_gt
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri




def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm




def get_parameter_groups(
    model, weight_decay, layer_decay=1.0, skip_list=(), no_lr_scale_list=[]
):
    parameter_group_names = {}
    parameter_group_vars = {}
    enc_depth, dec_depth = None, None
    # prepare layer decay values
    assert layer_decay == 1.0 or 0.0 < layer_decay < 1.0
    if layer_decay < 1.0:
        enc_depth = model.enc_depth
        dec_depth = model.dec_depth if hasattr(model, "dec_blocks") else 0
        num_layers = enc_depth + dec_depth
        layer_decay_values = list(
            layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)
        )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        # Assign weight decay values
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            if "enc_blocks" in name:
                group_name = "no_decay_enc_blocks"
            else:
                group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            if "enc_blocks" in name:
                group_name = "decay_enc_blocks"
            else:
                group_name = "decay"
            this_weight_decay = weight_decay
        
        if "decoder_token" in name:
            group_name = "decoder_token"
            this_weight_decay = 0.0

        # Assign layer ID for LR scaling
        if layer_decay < 1.0:
            skip_scale = False
            layer_id = _get_num_layer_for_vit(name, enc_depth, dec_depth)
            group_name = "layer_%d_%s" % (layer_id, group_name)
            if name in no_lr_scale_list:
                skip_scale = True
                group_name = f"{group_name}_no_lr_scale"
        else:
            layer_id = 0
            skip_scale = True

        if group_name not in parameter_group_names:
            if not skip_scale:
                scale = layer_decay_values[layer_id]
            else:
                scale = 1.0

            if "enc_blocks" in group_name:
                scale *= 1.0
            
            if "decoder_token" in group_name:
                scale *= 1e-2
            
            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def _get_num_layer_for_vit(var_name, enc_depth, dec_depth):
    if var_name in ("cls_token", "mask_token", "pos_embed", "global_tokens"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("enc_blocks"):
        layer_id = int(var_name.split(".")[1])
        return layer_id + 1
    elif var_name.startswith("decoder_embed") or var_name.startswith(
        "enc_norm"
    ):  # part of the last black
        return enc_depth
    elif var_name.startswith("dec_blocks"):
        layer_id = int(var_name.split(".")[1])
        return enc_depth + layer_id + 1
    elif var_name.startswith("dec_norm"):  # part of the last block
        return enc_depth + dec_depth
    elif any(var_name.startswith(k) for k in ["head", "prediction_head"]):
        return enc_depth + dec_depth + 1
    else:
        raise NotImplementedError(var_name)


def is_main_process(accelerator: Accelerator):
    return accelerator.is_main_process


def save_on_master(accelerator: Accelerator, *args, **kwargs):
    if is_main_process(accelerator):
        torch.save(*args, **kwargs)



class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values."""

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self, accelerator: Accelerator):
        """Synchronize the count and total across all processes."""
        if accelerator.num_processes == 1:
            return
        t = torch.tensor(
            [self.count, self.total], dtype=torch.float64, device=accelerator.device
        )
        accelerator.wait_for_everyone()
        accelerator.reduce(t, reduction="sum")
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )




def save_glb_scene_with_cleanup(predictions, batch, glbscene, output_dir, suffix=None):
    """
    Save GLB scene to viz subdirectory and maintain only latest 10 files.
    
    Args:
        glbscene: The trimesh scene to save
        args: Training arguments containing output_dir
        step: Current training step
        epoch: Current epoch
    """
    # Create viz subdirectory
    viz_dir = output_dir
    os.makedirs(viz_dir, exist_ok=True)
    
    # Generate filename with timestamp and step info
    if suffix is None:
        filename = f"glbscene.glb"
    else:
        filename = f"glbscene{suffix}.glb"
    filepath = os.path.join(viz_dir, filename)
    
    # Save the GLB scene
    glbscene.export(file_obj=filepath)
    print(f"Saved GLB scene to: {filepath}")

    # save npz file
    if "shape" in predictions:
        posterior_path = f"glbscene.npz"
        posterior_path = os.path.join(viz_dir, posterior_path)
        np.savez(posterior_path, 
            posterior=predictions["shape"],
            posterior_gt=batch['shape_tokens'][0].detach().cpu().numpy())
    
    return filepath
    
    # Clean up old files, keeping only the latest 10




def process_and_generate_glbscene(predictions, batch, output_dir, viz_gt=True, use_HY_token=False, prediction_mode="Predicted Pointmap"):
    """
    Processes model predictions, converts pose encoding, generates world points, and creates a GLB scene.
    Returns the generated glbscene.
    """

    # Create a copy of predictions to avoid modifying the original
    predictions_copy = { k: (v.detach().clone() if isinstance(v, torch.Tensor) else v) for k, v in predictions.items() }

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    if use_HY_token:
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions_copy["pose_enc_HY"], batch["images"].shape[-2:])
        # only use first frame prediction (apply transform to VGGT extrinsic & repeat scale)
        # scale predictions
        scale_HY = predictions_copy["scale_HY"].exp()
        extrinsic[:,:,:3,3] = extrinsic[:,:,:3,3] * scale_HY.unsqueeze(-1)
        predictions_copy["depth"] = predictions_copy["depth"] * scale_HY.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        predictions_copy["world_points"] = predictions_copy["world_points"] * scale_HY.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    else:
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions_copy["pose_enc"], batch["images"].shape[-2:])
    predictions_copy["extrinsic"] = extrinsic
    predictions_copy["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions_copy.keys():
        if isinstance(predictions_copy[key], torch.Tensor):
            predictions_copy[key] = predictions_copy[key].cpu().float().numpy().squeeze(0)  # remove batch dimension

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions_copy["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions_copy["extrinsic"], predictions_copy["intrinsic"])
    predictions_copy["world_points_from_depth"] = world_points

    ## for debug
    ## Save world points visualization
    ## Create output directory if it doesn't exist
    #if use_canon:
        ## Create point cloud visualization
        ## Get unique colors for each camera viewpoint
        ## Combine all points with different colors per camera

        ## Combine points and colors
        ## Create and save point cloud visualization
        ## Save both .ply file and screenshot

        ## Save depth map visualization
            ## Normalize depth to 0-1 range
            
            ## Convert to PIL image and save
            
            ## Save depth map as PNG

    glbscene, pred_vertices_3d, gt_vertices_3d, gt_np = predictions_to_glb_w_gt(
        predictions_copy,
        gt=batch if viz_gt else None,
        conf_thres=50.0,
        filter_by_frames="All",
        mask_black_bg=True,
        mask_white_bg=True,
        show_cam=True,
        mask_sky=False,
        target_dir=output_dir,
        prediction_mode=prediction_mode,
        use_HY_token=use_HY_token
    )

    return glbscene, pred_vertices_3d, gt_vertices_3d, predictions_copy, gt_np

#@ torch.no_grad()
    #"""
    #Extract mesh from trained SDF model using marching cubes
    #"""
    #if is_train:
    #else:

    ## Generate filename with timestamp and step info

    #if epoch>200:
    #else:
    #output = hierarchical_extract_geometry(
        #geometric_func,
        #model.device,
    #)




        


