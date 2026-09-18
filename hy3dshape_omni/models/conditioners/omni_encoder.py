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

from typing import Optional, Union, List
import math
import random
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import Dinov2Model
import os
import torch
import yaml
from collections import Counter
from huggingface_hub import snapshot_download

import importlib
from omegaconf import OmegaConf, DictConfig, ListConfig

def get_config_from_file(config_file: str) -> Union[DictConfig, ListConfig]:
    config_file = OmegaConf.load(config_file)

    if 'base_config' in config_file.keys():
        if config_file['base_config'] == "default_base":
            base_config = OmegaConf.create()
            # base_config = get_default_config()
        elif config_file['base_config'].endswith(".yaml"):
            base_config = get_config_from_file(config_file['base_config'])
        else:
            raise ValueError(f"{config_file} must be `.yaml` file or it contains `base_config` key.")

        config_file = {key: value for key, value in config_file if key != "base_config"}

        return OmegaConf.merge(base_config, config_file)

    return config_file

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)

    # Configs inherited from Hunyuan3D-Omni name their targets under `hy3dshape`;
    # this package vendors them as `hy3dshape_omni`, so rewrite the module path.
    module = module.replace('hy3dshape', 'hy3dshape_omni')

    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_obj_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    return get_obj_from_str(config["target"])

def instantiate_from_config(config, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    cls = get_obj_from_str(config["target"])

    params = config.get("params", dict())
    # params.update(kwargs)
    # instance = cls(**params)
    kwargs.update(params)
    instance = cls(**kwargs)

    return instance

def fps(
    src: torch.Tensor,
    batch: Optional[Tensor] = None,
    ratio: Optional[Union[Tensor, float]] = None,
    random_start: bool = True,
    batch_size: Optional[int] = None,
    ptr: Optional[Union[Tensor, List[int]]] = None,
):
    src = src.float()
    from torch_cluster import fps as fps_fn
    output = fps_fn(src, batch, ratio, random_start, batch_size, ptr)
    return output


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor):
        mask = torch.zeros_like(x).bool()
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)
        return pos


class DinoImageEncoder(nn.Module):
    def __init__(
        self,
        version="facebook/dinov2-large",
        trainable=False,
        image_size=224,
        use_cls_token=True,
        use_pos_embed=False,
        zero_out_background=False,
        mask_resize_mode='bicubic',
        **kwargs,
    ):
        super().__init__()
        self.model = Dinov2Model.from_pretrained(version)
        self.use_cls_token = use_cls_token
        self.use_pos_embed = use_pos_embed
        self.zero_out_background = zero_out_background
        self.mask_resize_mode = mask_resize_mode
        self.image_size = image_size
        self.setup_transform(image_size)
        if not trainable:
            self.model.eval()
            self.model.requires_grad_(False)

        if self.use_pos_embed:
            self.pos_embed = PositionEmbeddingSine(self.model.config.hidden_size // 2)

        if self.zero_out_background and self.use_pos_embed:
            raise ValueError("Cannot use zero_out_background and use_pos_embed at the same time")

    def setup_transform(self, image_size):
        print(f"Image size: {image_size}")
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.CenterCrop(image_size),  # crop a (224, 224) square
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.size = image_size // 14
        self.patch_nums = (image_size // 14) ** 2
        if self.use_cls_token:
            self.patch_nums += 1

    def expand_mask_to_bbox(self, masks):
        bs = masks.shape[0]
        expanded_masks = torch.zeros_like(masks)
        for i in range(bs):
            mask = masks[i, 0]
            non_zero_indices = torch.nonzero(mask, as_tuple=False)
            if non_zero_indices.numel() > 0:
                y_min, x_min = torch.min(non_zero_indices, dim=0)[0]
                y_max, x_max = torch.max(non_zero_indices, dim=0)[0]
                expanded_masks[i, 0, y_min:y_max + 1, x_min:x_max + 1] = 1.0
        return expanded_masks

    def forward(self, image, dropout_mask=None, value_range=(-1, 1), mask=None):
        if value_range is not None:
            low, high = value_range
            image = (image - low) / (high - low)

        inputs = self.transform(image)
        outputs = self.model(inputs)

        last_hidden_state = outputs.last_hidden_state
        if not self.use_cls_token:
            last_hidden_state = last_hidden_state[:, 1:, :]

        if self.use_pos_embed:
            if self.use_cls_token:
                raise NotImplementedError
            B, N, C = last_hidden_state.shape
            pos_embed = self.pos_embed(last_hidden_state[:, :, 0].reshape(B, self.size, self.size))
            pos_embed = pos_embed.reshape(B, N, C)
            last_hidden_state = last_hidden_state + pos_embed.to(last_hidden_state.device,
                                                                 dtype=last_hidden_state.dtype)

        if self.zero_out_background:
            if mask is None:
                assert self.training is False, "mask should be provided in training mode"
                print("Warning: mask is not provided, use mask compute from image")
                image_np = image.detach().cpu().numpy()
                image_np = np.transpose(image_np, (0, 2, 3, 1))[0]
                image_np = (image_np * 255).astype(np.uint8)
                mask = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                mask = (mask < 250).astype(np.uint8)
                mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
                mask = torch.from_numpy(mask).to(self.model.device, dtype=torch.bool)
            else:
                mask = (mask > 0) * 1.0
                if self.mask_resize_mode == 'maxpooling':
                    mask = F.interpolate(mask, size=(self.image_size, self.image_size), mode='nearest')
                    mask = F.max_pool2d(mask, kernel_size=14, stride=14)
                elif self.mask_resize_mode == 'bbox':
                    mask = self.expand_mask_to_bbox(mask)
                    mask = F.interpolate(mask, size=(self.image_size, self.image_size), mode='nearest')
                    mask = F.max_pool2d(mask, kernel_size=14, stride=14)
                else:
                    mask = F.interpolate(mask, size=(self.size, self.size), mode=self.mask_resize_mode)

            mask = mask.to(dtype=last_hidden_state.dtype)
            mask_flatten = mask.reshape(mask.shape[0], -1, 1)
            if not self.use_cls_token:
                last_hidden_state = last_hidden_state * mask_flatten
            else:
                new_hidden_state = last_hidden_state[:, 1:, :]
                new_hidden_state = new_hidden_state * mask_flatten
                last_hidden_state = torch.cat([last_hidden_state[:, 0:1, :], new_hidden_state], dim=1)

        outputs = {'dino': {'last_hidden_state': last_hidden_state}}

        if dropout_mask is not None:
            outputs = self.maskout(outputs, dropout_mask)

        return outputs

    def maskout(self, outputs, mask):
        bsz = mask.shape[0]
        mask = mask.reshape(bsz, 1, 1).to(self.model.device)
        mask = torch.logical_not(mask)
        last_hidden_state = outputs['dino']['last_hidden_state'] * mask
        return {'dino': {
            'last_hidden_state': last_hidden_state,
        }}

    def unconditional_embedding(self, batch_size):
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        zero = torch.zeros(
            batch_size,
            self.patch_nums,
            self.model.config.hidden_size,
            device=device,
            dtype=dtype,
        )
        return {'dino': {
            'last_hidden_state': zero,
        }}


class ModLN(nn.Module):
    def __init__(self, inner_dim: int, mod_dim: int = 1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mod_dim, inner_dim * 2),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x:torch.Tensor, condition:torch.Tensor):
        '''
        x: [N, M, C_in], M: num of tokens
        condition: [N, C_mod]
        '''
        shift, scale = self.mlp(condition).unsqueeze(1).chunk(2, dim=-1)
        return x * (1 + scale) + shift


class DinoEncoder(nn.Module):
    def __init__(
        self,
        dino_image_encoder_version,
        drop_image_dino_rate=0.0,
    ):
        super().__init__()
        self.dino_image_encoder = DinoImageEncoder(version=dino_image_encoder_version)
        self.drop_image_dino_rate = drop_image_dino_rate
        self.disable_drop = False

    def forward(self, image, text):
        outputs = {}

        if self.disable_drop:
            dino_mask = None
        else:
            random_p = torch.rand(len(image), device='cuda')
            dino_mask = random_p < self.drop_image_dino_rate

        dino_outputs = self.dino_image_encoder(image, dropout_mask=dino_mask)
        outputs.update(dino_outputs)

        return outputs

    def unconditional_embedding(self, batch_size):
        outputs = {}

        dino_outputs = self.dino_image_encoder.unconditional_embedding(batch_size)
        outputs.update(dino_outputs)

        return outputs


class SingleImageEncoder(nn.Module):
    def __init__(
        self,
        image_encoder,
        drop_ratio=0.0,
    ):
        super().__init__()
        self.image_encoder = instantiate_from_config(image_encoder)
        self.drop_ratio = drop_ratio
        self.disable_drop = False

    def setup(self, image_size=224):
        if hasattr(self.image_encoder, 'setup_transform'):
            self.image_encoder.setup_transform(image_size=image_size)

    def forward(self, image=None, text=None, mask=None):
        if self.disable_drop:
            dropout_mask = None
        else:
            random_p = torch.rand(len(image), device='cuda')
            dropout_mask = random_p < self.drop_ratio

        outputs = self.image_encoder(image, dropout_mask=dropout_mask, mask=mask)
        return outputs

    def unconditional_embedding(self, batch_size):
        outputs = self.image_encoder.unconditional_embedding(batch_size)
        return outputs


class OmniEncoder(nn.Module):
    def __init__(
        self,
        image_encoder,
        image_size=224,
        resolutions=[512, 1024, 2048],
        voxel_resolution=16,
        width=1024,
        re_sample=True,
        random_noise=False,
        noise_ratio=0.0,
        noise_scales=[0.0],
        drop_point=False,
        drop_ratio=0.0,
        num_freqs=8,
        include_pi=True,
    ):
        super().__init__()
        self.drop_ratio = drop_ratio
        self.disable_drop = False
        self.image_encoder = instantiate_from_config(image_encoder)
        self.image_encoder.eval()
        self.image_encoder.requires_grad_(False)

        self.cond_signal_embedding = nn.Embedding(4, 8)
        self.cond_signal_linear = nn.Linear(8, width)

        # parameter for point
        self.voxel_resolution = voxel_resolution
        self.resolutions = resolutions
        self.random_noise = random_noise
        self.noise_scales = noise_scales
        self.noise_ratio = noise_ratio
        self.re_sample = re_sample
        self.drop_point = drop_point
        self.drop_ratio = drop_ratio
        self.include_pi = include_pi
        from ..modules.embedder import FourierEmbedder
        self.pe = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)
        self.linear = nn.Sequential(
                nn.Linear(self.pe.get_dims(6), width),
                nn.RMSNorm(width),
                nn.GELU()
        )

    def setup(self, image_size=224):
        if hasattr(self.image_encoder, 'setup_transform'):
            self.image_encoder.setup_transform(image_size=image_size)
    
    def generate_voxel(self, pc):
        '''
        Quantise the point cloud onto a 16*16*16 grid
        '''
        device, dtype = pc.device, pc.dtype
        B, N, D = pc.shape
        assert D == 3, "point cloud must be 3-dimensional" # pc: -1 ~ 1
        
        resolution = self.voxel_resolution
        points_norm = (pc + 1) / 2  # (B, N, 3) 0 ~ 1
        voxels = (points_norm * resolution).floor().long()  # (B, N, 3)
        voxels = torch.clamp(voxels, 0, resolution - 1)
    
        sampled_voxels_batch = []
        for b in range(B):
            vox_b = voxels[b]  # (N, 3)
            linear_idx = vox_b[:, 0] + vox_b[:, 1] * resolution + vox_b[:, 2] * resolution * resolution
            unique_idx = torch.unique(linear_idx)
    
            # back to a 3D voxel index
            z = unique_idx // (resolution * resolution)
            y = (unique_idx % (resolution * resolution)) // resolution
            x = unique_idx % resolution
            unique_voxels = torch.stack([x, y, z], dim=1).float()  # (M, 3)
    
            # mapping back to [-1,1], get center of voxel
            # range of voxel idx is [0, resolution-1]
            # voxel size = 2 / resolution
            voxel_size = 2.0 / resolution
            voxel_centers = unique_voxels * voxel_size + voxel_size / 2 - 1  # (M, 3)
            sampled_voxels_batch.append(voxel_centers)
    
        # padding to same length for batch process
        max_voxels = max([v.shape[0] for v in sampled_voxels_batch])
        padded_voxels = []
        for v in sampled_voxels_batch:
            pad_len = max_voxels - v.shape[0]
            if pad_len > 0:
                pad = torch.zeros(pad_len, 3, device=pc.device, dtype=pc.dtype)
                v = torch.cat([v, pad], dim=0)
            padded_voxels.append(v.unsqueeze(0))  # (1, max_voxels, 3)
    
        sampled_voxels = torch.cat(padded_voxels, dim=0)  # (B, max_voxels, 3)

        return sampled_voxels.to(device=device, dtype=dtype)
    
    def bbox_to_corners(self, bbox):
        """
        PyTorch version: turn a bbox (B,1,3) into its 8 corner coordinates (range [-1,1])
        
        Args:
            bbox: torch.Tensor, shape (B,1,3), holding [length, height, width] (range 0~1)
        
        Returns:
            corners: torch.Tensor, shape (B,8,3), the xyz of each bbox's 8 corners
        """
        B = bbox.shape[0]
        half_dims = bbox / 2  # (B,1,3)
        
        signs = torch.tensor([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ], dtype=torch.float32, device=bbox.device)  # (8,3)
        
        corners = half_dims * signs.unsqueeze(0)  # (B,8,3)
        
        return corners

    def forward(self, image, surface=None, mask=None, pose=None, bbox=None, point=None, voxel=None, vggt_encoder_tokens=None, **kwargs):
        if self.disable_drop:
            dropout_mask = None
        else:
            random_p = torch.rand(len(image), device='cuda')
            dropout_mask = random_p < self.drop_ratio

        image_cond = self.image_encoder(image, dropout_mask=dropout_mask, mask=mask)['dino']['last_hidden_state']
        if pose is not None:
            cond = self.linear(self.pe(pose))
            cond_signal = self.cond_signal_embedding(torch.tensor([0], device=cond.device))
            cond_signal = self.cond_signal_linear(cond_signal)
            cond_signal = cond_signal.unsqueeze(0).repeat(len(image), 10, 1)
            cond = torch.cat([image_cond, cond, cond_signal], dim=1)
            sampled_point = pose[..., :3]

        elif bbox is not None:
            cond = self.linear(self.pe(bbox.repeat(1, 1, 2)))
            cond_signal = self.cond_signal_embedding(torch.tensor([1], device=cond.device))
            cond_signal = self.cond_signal_linear(cond_signal)
            cond_signal = cond_signal.unsqueeze(0).repeat(len(image), 10, 1)
            cond = torch.cat([image_cond, cond, cond_signal], dim=1)
            sampled_point = self.bbox_to_corners(bbox)

        elif voxel is not None:
            voxel = self.generate_voxel(voxel[..., :3])
            cond = self.linear(self.pe(voxel.repeat(1, 1, 2)))
            cond_signal = self.cond_signal_embedding(torch.tensor([2], device=cond.device))
            cond_signal = self.cond_signal_linear(cond_signal)
            cond_signal = cond_signal.unsqueeze(0).repeat(len(image), 10, 1)
            cond = torch.cat([image_cond, cond, cond_signal], dim=1)
            sampled_point = voxel[..., :3]
        
        elif vggt_encoder_tokens is not None:
            cond = image_cond
            sampled_point = torch.zeros(image_cond.shape[0], 1, 3, device=image_cond.device, dtype=image_cond.dtype) 

        elif point is not None:
            cond = self.linear(self.pe(point.repeat(1, 1, 2)))
            cond_signal = self.cond_signal_embedding(torch.tensor([3], device=cond.device))
            cond_signal = self.cond_signal_linear(cond_signal)
            cond_signal = cond_signal.unsqueeze(0).repeat(len(image), 10, 1)
            cond = torch.cat([image_cond, cond, cond_signal], dim=1)
            sampled_point = point[..., :3]
        else:
            raise ValueError(f"pose, bbox, voxel, point must be one of them")

        outputs = {
            'main': cond,
            'cond_point': sampled_point
        }
        return outputs
    
    def images_dino_encoder(self, images, masks=None) :
        B, S, C, H, W = images.shape

        if self.disable_drop:
            dropout_mask = None
        else:
            random_p = torch.rand(len(images), device='cuda')
            dropout_mask = random_p < self.drop_ratio
            dropout_mask = dropout_mask[:, None].expand(B, S).reshape(B*S)
        
        images = images.reshape(B*S, C, H, W)
        if masks is not None:
            masks = masks.reshape(B*S, H, W)

        image_cond = self.image_encoder(images, dropout_mask=dropout_mask, mask=masks)['dino']['last_hidden_state']

        _, N, D = image_cond.shape

        return image_cond.reshape(B, S, N, D)


    def cond_point_encoder(self, point):
        cond = self.linear(self.pe(point.repeat(1, 1, 2)))
        cond_signal = self.cond_signal_embedding(torch.tensor([3], device=cond.device))
        cond_signal = self.cond_signal_linear(cond_signal)
        cond_signal = cond_signal.unsqueeze(0).repeat(point.shape[0], 10, 1)
        cond = torch.cat([cond, cond_signal], dim=1)
        sampled_point = point[..., :3]
        return cond, sampled_point

    def unconditional_embedding(self, batch_size):
        outputs = self.image_encoder.unconditional_embedding(batch_size)
        return outputs

    @classmethod
    def from_pretrained(cls,
                    model_path,
                    variant=None,
                    device='cuda',
                    dtype=torch.float16,
                    subfolder='cond_encoder',
                    resume_download=False,
                    force_download=False,
                    revision="main",
                    **kwargs):
        """
        Load a pretrained OmniEncoder from a model path.
        
        Args:
            model_path: Path to the model directory (local or HuggingFace repo_id)
            variant: Optional variant name (e.g., 'ema' for EMA weights)
            device: Device to load the model on
            dtype: Data type for the model
            resume_download: Whether to resume interrupted downloads
            force_download: Whether to force re-download
            revision: The specific model version to use
            **kwargs: Additional arguments
        
        Returns:
            OmniEncoder instance with loaded weights
        """
        # Handle local vs remote model path
        if os.path.exists(model_path):
            print(f'Loading OmniEncoder from local path: {model_path}')
        else:
            repo_id = model_path
            base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
            model_path = os.path.expanduser(os.path.join(base_dir, repo_id))
            print(f'Loading OmniEncoder from huggingface cache: {model_path}')

            if not os.path.exists(model_path):
                print(f'Not Found {model_path}')
                print(f'Downloading model from huggingface: {repo_id}')
            
                from huggingface_hub import snapshot_download
                path = snapshot_download(
                    repo_id=repo_id,
                    local_dir=model_path,
                    local_dir_use_symlinks=False,
                    resume_download=resume_download,
                    force_download=force_download,
                    revision=revision,
                )
                print('Downloaded to:', path)
        
        # Load config
        config_path = os.path.join(model_path, subfolder, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update model params with kwargs
        model_kwargs = config.get('params', config)
        model_kwargs.update(kwargs)
        
        # Instantiate model
        model = cls(**model_kwargs)

        # Load weights
        ckpt_name = 'pytorch_model.bin'
        # Handle EMA variant if requested
        if variant == 'ema':
            ckpt_name = 'pytorch_model_ema.bin'
            ckpt_path = os.path.join(model_path, subfolder, ckpt_name)
            if not os.path.exists(ckpt_path):
                print(f'EMA weights not found, falling back to standard weights')
                ckpt_name = 'pytorch_model.bin'
                ckpt_path = os.path.join(model_path, subfolder, ckpt_name)

        ckpt_path = os.path.join(model_path, subfolder, ckpt_name)
        
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint from: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location='cpu')
            
            # Handle nested 'model' key in state dict
            if 'model' in state_dict:
                state_dict = state_dict['model']
            
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            
            print(f"Loaded {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
            if len(missing) > 0:
                from collections import Counter
                print(f"Missing Keys: {Counter([s.split('.')[0] for s in missing])}")
            if len(unexpected) > 0:
                from collections import Counter
                print(f"Unexpected Keys: {Counter([s.split('.')[0] for s in unexpected])}")
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        
        # relearn the omni encoder
        #model.linear = nn.Sequential(
            #nn.Linear(model.pe.get_dims(6), 1024),
            #nn.RMSNorm(1024),
        #)

        # Set model properties and move to device
        model = model.to(device=device, dtype=dtype)
        
        print(f'Successfully loaded OmniEncoder')
        return model



class CondPointEncoder_MultiView(nn.Module):
    def __init__(self, omni_encoder: OmniEncoder, conf_inject: bool = False):
        super().__init__()
        
        # Clone the embeddings and layers to create independent trainable copies
        # Get the architecture details from omni_encoder
        embedding_dim = omni_encoder.cond_signal_embedding.embedding_dim
        num_embeddings = omni_encoder.cond_signal_embedding.num_embeddings
        
        # Create new embedding and linear layers with same architecture
        self.cond_signal_embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.cond_signal_linear = nn.Linear(embedding_dim, omni_encoder.cond_signal_linear.out_features)
        
        # Clone the linear layer (Sequential module)
        self.linear = nn.Sequential(
            nn.Linear(omni_encoder.linear[0].in_features, omni_encoder.linear[0].out_features),
            nn.RMSNorm(omni_encoder.linear[1].normalized_shape[0]),
            nn.GELU()
        )

        self.vggt_linear = nn.Sequential(
            nn.Linear(8192, omni_encoder.linear[0].out_features),
            #nn.Linear(2048, omni_encoder.linear[0].out_features),
            nn.RMSNorm(omni_encoder.linear[1].normalized_shape[0]),
            nn.GELU()
        )

        self.camera_tokens_linear = nn.Sequential(
            nn.Linear(2048,omni_encoder.linear[1].normalized_shape[0]),
            nn.RMSNorm(omni_encoder.linear[1].normalized_shape[0]),
            nn.GELU()
        )

        self.dino_features_linear = None
        #self.dino_features_linear = nn.Sequential(
            #nn.Linear(1024,omni_encoder.linear[1].normalized_shape[0]),
            #nn.RMSNorm(omni_encoder.linear[1].normalized_shape[0]),
        #)
        
        # Clone the positional encoder
        from ..modules.embedder import FourierEmbedder
        self.pe = FourierEmbedder(
            num_freqs=omni_encoder.pe.num_freqs,
            include_pi=omni_encoder.include_pi
        )
        
        # Copy the weights from omni_encoder (clone and detach to create independent copies)
        with torch.no_grad():
            self.cond_signal_embedding.weight.copy_(omni_encoder.cond_signal_embedding.weight.clone())
            self.cond_signal_linear.weight.copy_(omni_encoder.cond_signal_linear.weight.clone())
            self.cond_signal_linear.bias.copy_(omni_encoder.cond_signal_linear.bias.clone())
            
            # Copy linear layer weights
            self.linear[0].weight.copy_(omni_encoder.linear[0].weight.clone())
            self.linear[0].bias.copy_(omni_encoder.linear[0].bias.clone())
            self.linear[1].weight.copy_(omni_encoder.linear[1].weight.clone())
        

        # === FiLM head (only used if conf_inject=True) ===
        self.conf_inject = conf_inject
        if conf_inject:
            pe_dim = omni_encoder.linear[0].in_features  # input dim to the first linear equals PE output dim
            # tiny MLP to produce gamma and beta for each channel
            self.conf_head = nn.Sequential(
                nn.Linear(1, pe_dim // 4),
                nn.GELU(),
                nn.Linear(pe_dim // 4, pe_dim * 2)  # [gamma, beta]
            )

            # Optional: Initialize gamma to near-zero so it starts as identity
            nn.init.zeros_(self.conf_head[-1].weight)
            nn.init.zeros_(self.conf_head[-1].bias)
        
        
        # Ensure all parameters are trainable
        self.cond_signal_embedding.requires_grad_(True)
        self.cond_signal_linear.requires_grad_(True)
        self.linear.requires_grad_(True)
        self.vggt_linear.requires_grad_(True)
        self.camera_tokens_linear.requires_grad_(True)
        if self.conf_inject:
            self.conf_head.requires_grad_(True)
        if self.dino_features_linear is not None:
            self.dino_features_linear.requires_grad_(True)
    
    def forward(self, point,  conf=None, dino_features=None, point_camera_tokens=None, vggt_tokens=None):
        """
        Encodes point cloud conditioning following the cond_point_encoder logic.
        
        Args:
            point: Point cloud tensor of shape (B, N, 3)
        
        Returns:
            cond: Encoded conditioning tensor
            sampled_point: The first 3 dimensions of the input points
        """
        # Apply positional encoding and linear transformation
        pe_feat = self.pe(point.repeat(1, 1, 2))

        if self.conf_inject:
            if conf.dim() == 2:  # (B,N) -> (B,N,1)
                conf = conf.unsqueeze(-1)
            # Normalize conf if useful (optional):
            gb = self.conf_head(conf)     # (B, N, 2C)
            C = pe_feat.shape[-1]
            gamma, beta = gb.split(C, dim=-1)  # each (B, N, C)
            # FiLM modulation; small scale on gamma helps stability
            pe_feat = pe_feat * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * torch.tanh(beta)
        
        cond = self.linear(pe_feat)
        cond_vggt = self.vggt_linear(vggt_tokens)
        cond_camera_tokens = self.camera_tokens_linear(point_camera_tokens)

        # Generate condition signal (signal type 3 for point cloud)
        cond_signal = self.cond_signal_embedding(torch.tensor([3], device=cond.device))
        cond_signal = self.cond_signal_linear(cond_signal)
        cond_signal = cond_signal.unsqueeze(0).repeat(point.shape[0], 10, 1)
        
        # Concatenate point encoding with signal
        cond_dino = cond_vggt + cond_camera_tokens + dino_features

        cond_enhanced = cond

        cond = torch.cat([cond_dino, cond_enhanced, cond_signal], dim=1)
        
        # Extract xyz coordinates
        sampled_point = point[..., :3]
        
        return cond, sampled_point


