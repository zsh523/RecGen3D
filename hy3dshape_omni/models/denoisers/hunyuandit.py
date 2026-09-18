# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.
# Modified by the RecGen3D authors, 2026.

import os
import yaml
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Union
from torch import Tensor
import deepspeed

from .moe_layers import MoEBlock
from ...utils import logger, synchronize_timer, smart_load_model

from .rope import RotaryPositionEmbedding2D

class ChunkedLayerNorm(nn.Module):
    def __init__(self, num_chunks: int, chunk_dim: int, eps: float = 1e-6):
        super().__init__()
        self.num_chunks = num_chunks
        self.chunk_dim = chunk_dim
        # One LN shared across all chunks
        self.ln = nn.LayerNorm(chunk_dim, eps=eps)

    def forward(self, x):
        # x: [B, N, num_chunks*chunk_dim]
        B, N, D = x.shape
        assert D == self.num_chunks * self.chunk_dim

        # reshape -> [B, N, num_chunks, chunk_dim]
        x = x.view(B, N, self.num_chunks, self.chunk_dim)
        # apply LN per chunk
        x = self.ln(x)
        # back to [B, N, num_chunks*chunk_dim]
        return x.view(B, N, D)

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: Union[float, Tensor] = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)


class Timesteps(nn.Module):
    def __init__(self,
                 num_channels: int,
                 downscale_freq_shift: float = 0.0,
                 scale: int = 1,
                 max_period: int = 10000
                 ):
        super().__init__()
        self.num_channels = num_channels
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def forward(self, timesteps):
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        embedding_dim = self.num_channels
        half_dim = embedding_dim // 2
        exponent = -math.log(self.max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = self.scale * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, cond_proj_dim=None, out_size=None):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, frequency_embedding_size, bias=True),
            nn.GELU(),
            nn.Linear(frequency_embedding_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, frequency_embedding_size, bias=False)

        self.time_embed = Timesteps(hidden_size)

    def forward(self, t, condition):

        t_freq = self.time_embed(t).type(self.mlp[0].weight.dtype)

        # t_freq = timestep_embedding(t, self.frequency_embedding_size).type(self.mlp[0].weight.dtype)
        if condition is not None:
            t_freq = t_freq + self.cond_proj(condition)

        t = self.mlp(t_freq)
        t = t.unsqueeze(dim=1)
        return t


class MLP(nn.Module):
    def __init__(self, *, width: int):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class GroupedFeatureFusion(nn.Module):
    def __init__(self, layer_dims=(2048, 2048, 2048, 2048, 1024), out_dim=1024,
                 bottleneck_ratio=2.0, use_tokenwise_gate=False, 
                 dropout=0.0, ffn_ratio=4.0):
        super().__init__()
        self.layer_dims = list(layer_dims)
        self.num_src = len(self.layer_dims)
        self.out_dim = out_dim
        self.use_tokenwise_gate = use_tokenwise_gate
        
        bottleneck = int(out_dim * bottleneck_ratio)
        
        # per-source projection
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d, eps=1e-6),
                nn.Linear(d, bottleneck, bias=True),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck, out_dim, bias=True),
            ) for d in self.layer_dims
        ])
        
        # gate
        if use_tokenwise_gate:
            self.gate_head = nn.Linear(out_dim, 1, bias=False)
            self.gate_temp = nn.Parameter(torch.ones(1))  # learnable temperature
        else:
            self.global_gate = nn.Parameter(torch.zeros(self.num_src))
        
        # residual path
        self.skip = nn.Linear(sum(self.layer_dims), out_dim, bias=False)
        
        # Post-processing
        self.post_norm = nn.LayerNorm(out_dim, eps=1e-6)
        ffn_hidden = int(out_dim * ffn_ratio)
        self.post_ffn = nn.Sequential(
            nn.Linear(out_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, out_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, y):
        B, T, D = y.shape
        assert D == sum(self.layer_dims)
        
        parts = torch.split(y, self.layer_dims, dim=-1)
        proj = [branch(pi) for branch, pi in zip(self.branches, parts)]
        proj = torch.stack(proj, dim=2)  # (B,T,S,out_dim)
        
        # gate
        if self.use_tokenwise_gate:
            logits = self.gate_head(proj).squeeze(-1)  # (B,T,S)
            gate = torch.softmax(logits / self.gate_temp.abs(), dim=2)
        else:
            gate = torch.softmax(self.global_gate, dim=0).view(1,1,self.num_src)
        
        fused = torch.sum(proj * gate.unsqueeze(-1), dim=2)
        
        # residual
        skip = self.skip(y)
        z = fused + skip
        
        # Post-processing
        z = self.post_norm(z)
        z = z + self.post_ffn(z)
        
        return z

class CrossAttention_lora_free(nn.Module):
    def __init__(
        self,
        qdim,
        kdim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        with_decoupled_ca=False,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        **kwargs,
    ):
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "self.qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim ** -0.5

        self.to_q_new = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.to_k_new = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.to_v_new = nn.Linear(kdim, qdim, bias=qkv_bias)

        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.out_proj_new = nn.Linear(qdim, qdim, bias=True)

        self.with_dca = with_decoupled_ca
        if self.with_dca:
            self.kv_proj_dca = nn.Linear(kdim, 2 * qdim, bias=qkv_bias)
            self.k_norm_dca = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
            self.dca_dim = decoupled_ca_dim
            self.dca_weight = decoupled_ca_weight
        
        self.rope = RotaryPositionEmbedding2D(frequency=100)

    def forward(self, x, y, ypos = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen1, hidden_dim) (where hidden_dim = num heads * head dim)
        y: torch.Tensor
            (batch, seqlen2, hidden_dim2)
        freqs_cis_img: torch.Tensor
            (batch, hidden_dim // 2), RoPE for image
        """
        b, s1, c = x.shape  # [b, s1, D]

        if self.with_dca:
            token_len = y.shape[1]
            context_dca = y[:, -self.dca_dim:, :]
            kv_dca = self.kv_proj_dca(context_dca).view(b, self.dca_dim, 2, self.num_heads, self.head_dim)
            k_dca, v_dca = kv_dca.unbind(dim=2)  # [b, s, h, d]
            k_dca = self.k_norm_dca(k_dca)
            y = y[:, :(token_len - self.dca_dim), :]

        _, s2, c = y.shape  # [b, s2, 1024]
        q = self.to_q_new(x)
        k = self.to_k_new(y)
        v = self.to_v_new(y)

        kv = torch.cat((k, v), dim=-1)
        split_size = kv.shape[-1] // self.num_heads // 2
        kv = kv.view(1, -1, self.num_heads, split_size * 2)
        k, v = torch.split(kv, split_size, dim=-1)

        q = q.view(b, s1, self.num_heads, self.head_dim)  # [b, s1, h, d]
        k = k.view(b, s2, self.num_heads, self.head_dim)  # [b, s2, h, d]
        v = v.view(b, s2, self.num_heads, self.head_dim)  # [b, s2, h, d]

        q = self.q_norm(q)
        k = self.k_norm(k)

        if ypos is not None:
            raise NotImplementedError("RoPE is not implemented for CrossAttention_lora_free")
            k = k.permute(0, 2, 1, 3)
            k = self.rope(k, ypos)
            k = k.permute(0, 2, 1, 3)

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=True
        ):
            q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.num_heads), (q, k, v))
            context = F.scaled_dot_product_attention(
                q, k, v
            ).transpose(1, 2).reshape(b, s1, -1)

        if self.with_dca:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=True
            ):
                k_dca, v_dca = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.num_heads),
                                   (k_dca, v_dca))
                context_dca = F.scaled_dot_product_attention(
                    q, k_dca, v_dca).transpose(1, 2).reshape(b, s1, -1)

            context = context + self.dca_weight * context_dca

        out = self.out_proj_new(context)  # context.reshape - B, L1, -1

        return out


class CrossAttention(nn.Module):
    def __init__(
        self,
        qdim,
        kdim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        with_decoupled_ca=False,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        **kwargs,
    ):
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "self.qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.to_k = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.to_v = nn.Linear(kdim, qdim, bias=qkv_bias)

        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(qdim, qdim, bias=True)

        self.with_dca = with_decoupled_ca
        if self.with_dca:
            self.kv_proj_dca = nn.Linear(kdim, 2 * qdim, bias=qkv_bias)
            self.k_norm_dca = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
            self.dca_dim = decoupled_ca_dim
            self.dca_weight = decoupled_ca_weight

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen1, hidden_dim) (where hidden_dim = num heads * head dim)
        y: torch.Tensor
            (batch, seqlen2, hidden_dim2)
        freqs_cis_img: torch.Tensor
            (batch, hidden_dim // 2), RoPE for image
        """
        b, s1, c = x.shape  # [b, s1, D]

        if self.with_dca:
            token_len = y.shape[1]
            context_dca = y[:, -self.dca_dim:, :]
            kv_dca = self.kv_proj_dca(context_dca).view(b, self.dca_dim, 2, self.num_heads, self.head_dim)
            k_dca, v_dca = kv_dca.unbind(dim=2)  # [b, s, h, d]
            k_dca = self.k_norm_dca(k_dca)
            y = y[:, :(token_len - self.dca_dim), :]

        _, s2, c = y.shape  # [b, s2, 1024]
        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)

        kv = torch.cat((k, v), dim=-1)
        split_size = kv.shape[-1] // self.num_heads // 2
        kv = kv.view(1, -1, self.num_heads, split_size * 2)
        k, v = torch.split(kv, split_size, dim=-1)

        q = q.view(b, s1, self.num_heads, self.head_dim)  # [b, s1, h, d]
        k = k.view(b, s2, self.num_heads, self.head_dim)  # [b, s2, h, d]
        v = v.view(b, s2, self.num_heads, self.head_dim)  # [b, s2, h, d]

        q = self.q_norm(q)
        k = self.k_norm(k)

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=True
        ):
            q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.num_heads), (q, k, v))
            context = F.scaled_dot_product_attention(
                q, k, v
            ).transpose(1, 2).reshape(b, s1, -1)

        if self.with_dca:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=True
            ):
                k_dca, v_dca = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.num_heads),
                                   (k_dca, v_dca))
                context_dca = F.scaled_dot_product_attention(
                    q, k_dca, v_dca).transpose(1, 2).reshape(b, s1, -1)

            context = context + self.dca_weight * context_dca

        out = self.out_proj(context)  # context.reshape - B, L1, -1

        return out


class Attention(nn.Module):
    """
    We rename some layer names to align with flash attention
    """

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.head_dim = self.dim // num_heads
        # This assertion is aligned with flash attention
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        qkv = torch.cat((q, k, v), dim=-1)
        split_size = qkv.shape[-1] // self.num_heads // 3
        qkv = qkv.view(1, -1, self.num_heads, split_size * 3)
        q, k, v = torch.split(qkv, split_size, dim=-1)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [b, h, s, d]
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [b, h, s, d]
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)  # [b, h, s, d]
        k = self.k_norm(k)  # [b, h, s, d]

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=True
        ):
            x = F.scaled_dot_product_attention(q, k, v)
            x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.out_proj(x)
        return x


class HunYuanDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        c_emb_size,
        num_heads,
        text_states_dim=1024,
        use_flash_attn=False,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        qk_norm_layer=nn.RMSNorm,
        with_decoupled_ca=False,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        init_scale=1.0,
        qkv_bias=True,
        skip_connection=True,
        timested_modulate=False,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
        vggt_text_states_dim=0,
        vggt_attn = False,
        vggt_attn_cross = True,
        vggt_feature_fusion_enabled = False,
        wan_style_timestep_modulation = False,
        pc_input = False,
        **kwargs,
    ):
        super().__init__()
        self.use_flash_attn = use_flash_attn
        use_ele_affine = True

        # ========================= Self-Attention =========================
        self.norm1 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)
        self.attn1 = Attention(hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
                               norm_layer=qk_norm_layer)

        # ========================= FFN =========================
        self.norm2 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)

        # ========================= Add =========================
        # Simply use add like SDXL.
        self.timested_modulate = timested_modulate
        if self.timested_modulate:
            self.default_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(c_emb_size, hidden_size, bias=True)
            )

        # ========================= Cross-Attention =========================
        self.attn2 = CrossAttention(hidden_size, text_states_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_norm=qk_norm, norm_layer=qk_norm_layer,
                                    with_decoupled_ca=with_decoupled_ca, decoupled_ca_dim=decoupled_ca_dim,
                                    decoupled_ca_weight=decoupled_ca_weight, init_scale=init_scale,
                                    )
        # ========================= VGGT Cross-Attention =========================
        if vggt_attn:
            self.vggt_attn_cross = vggt_attn_cross
            if vggt_attn_cross:
                if pc_input:
                    vggt_text_states_dim = 55
                if vggt_feature_fusion_enabled == True:
                    self.vggt_feature_fusion = GroupedFeatureFusion(layer_dims=(2048, 2048, 2048, 2048, 1024), out_dim=1024)
                    vggt_text_states_dim = 1024
                else:
                    self.vggt_feature_fusion = None
                self.attn_vggt = CrossAttention_lora_free(hidden_size, vggt_text_states_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                            qk_norm=qk_norm, norm_layer=qk_norm_layer,
                                            with_decoupled_ca=with_decoupled_ca, decoupled_ca_dim=decoupled_ca_dim,
                                            decoupled_ca_weight=decoupled_ca_weight, init_scale=init_scale,
                                            )
                self.attn_vggt_norm = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)
                self.attn_vggt_y_norm = norm_layer(vggt_text_states_dim, elementwise_affine=use_ele_affine, eps=1e-6)

                self.attn_vggt_linear = nn.Identity()

                if wan_style_timestep_modulation:
                    if hidden_size != vggt_text_states_dim:
                        raise ValueError(f"hidden_size: {hidden_size} should be equal to vggt_text_states_dim: {vggt_text_states_dim} when wan_style_timestep_modulation is True")
                    self.cross_attn_modulation = nn.Parameter(torch.randn(1, 3, hidden_size) / hidden_size**0.5)
            else:
                raise ValueError("vggt_attn_cross should be True")
                #self.attn_vggt = Attention(hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
                                   #norm_layer=qk_norm_layer)

                ## zero init
    
        self.norm3 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)

        if skip_connection:
            self.skip_norm = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None

        self.use_moe = use_moe
        if self.use_moe:
            print("using moe")
            self.moe = MoEBlock(
                hidden_size,
                num_experts=num_experts,
                moe_top_k=moe_top_k,
                dropout=0.0,
                activation_fn="gelu",
                final_dropout=False,
                ff_inner_dim=int(hidden_size * 4.0),
                ff_bias=True,
            )
        else:
            self.mlp = MLP(width=hidden_size)

    def forward(self, x, c=None, text_states=None, skip_value=None, vggt_text_states=None, vggt_text_states_pos=None, e=None):

        if self.skip_linear is not None:
            cat = torch.cat([skip_value, x], dim=-1)
            x = self.skip_linear(cat)
            x = self.skip_norm(x)

        # Self-Attention
        if self.timested_modulate:
            shift_msa = self.default_modulation(c).unsqueeze(dim=1)
            x = x + shift_msa

        attn_out = self.attn1(self.norm1(x))

        x = x + attn_out

        # Cross-Attention
        x = x + self.attn2(self.norm2(x), text_states)

        # VGGT Cross-Attention
        if vggt_text_states is not None:
            if self.vggt_attn_cross:
                if self.vggt_feature_fusion is None:
                    if e is not None:
                        e = (self.cross_attn_modulation + e).chunk(3, dim=1)
                        y = self.attn_vggt_y_norm(vggt_text_states)
                        y_modulated = y * (1 + e[1]) + e[0]
                        x = x + self.attn_vggt_linear(e[2] * self.attn_vggt(self.attn_vggt_norm(x), y_modulated, ypos=None))
                    else:
                        x = x + self.attn_vggt_linear(self.attn_vggt(self.attn_vggt_norm(x), self.attn_vggt_y_norm(vggt_text_states), ypos=None))
                else:
                    x = x + self.attn_vggt_linear(self.attn_vggt(self.attn_vggt_norm(x), self.attn_vggt_y_norm(self.vggt_feature_fusion(vggt_text_states)), ypos=None))
            else:
                raise ValueError("vggt_attn_cross should be True")
        else:
            pass

        # FFN Layer
        mlp_inputs = self.norm3(x)

        if self.use_moe:
            x = x + self.moe(mlp_inputs)
        else:
            x = x + self.mlp(mlp_inputs)

        return x


class AttentionPool(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, attention_mask=None):
        x = x.permute(1, 0, 2)  # NLC -> LNC
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).permute(1, 0, 2)
            global_emb = (x * attention_mask).sum(dim=0) / attention_mask.sum(dim=0)
            x = torch.cat([global_emb[None,], x], dim=0)

        else:
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class FinalLayer(nn.Module):
    """
    The final layer of HunYuanDiT.
    """

    def __init__(self, final_hidden_size, out_channels):
        super().__init__()
        self.final_hidden_size = final_hidden_size
        self.norm_final = nn.LayerNorm(final_hidden_size, elementwise_affine=True, eps=1e-6)
        self.linear = nn.Linear(final_hidden_size, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = x[:, 1:]
        x = self.linear(x)
        return x

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

class HunYuanDiTPlain(nn.Module):

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        device='cuda',
        dtype=torch.float16,
        variant=None,
        subfolder='model',
        resume_download=False,
        force_download=False,
        revision="main",
        **kwargs,
    ):
        # Check if model_path exists locally
        if os.path.exists(model_path):
            print(f'Loading model from local path: {model_path}')
        else:
            # Treat as HuggingFace repo_id
            repo_id = model_path
            base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
            model_path = os.path.expanduser(os.path.join(base_dir, repo_id))
            print(f'Loading model from huggingface cache: {model_path}')
            
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
                print(f'Downloaded to: {path}')
        
        # Load config
        config_path = os.path.join(model_path, subfolder, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract model params from config
        if 'model' in config:
            config = config['model']
        
        # Handle vggt_mode kwargs
        if kwargs.get('vggt_mode', None) is not None:
            if kwargs.get('vggt_mode', None) in ['additional', 'additional-dpt']:
                config['params']['vggt_attn'] = True
            elif kwargs.get('vggt_mode', None) == 'pc_input':
                config['params']['vggt_attn'] = True
                config['params']['pc_input'] = True
        
        # Update model params with kwargs
        model_kwargs = config.get('params', config)
        model_kwargs.update(kwargs)
        
        # Instantiate model
        model = cls(**model_kwargs)
        
        # Load checkpoint
        ckpt_name = 'pytorch_model.bin'
        if variant == 'ema':
            ckpt_name = 'pytorch_model_ema.bin'
            ckpt_path = os.path.join(model_path, subfolder, ckpt_name)
            if not os.path.exists(ckpt_path):
                print(f"EMA checkpoint not found, falling back to regular checkpoint")
                ckpt_name = 'pytorch_model.bin'
        
        ckpt_path = os.path.join(model_path, subfolder, ckpt_name)
        
        if os.path.exists(ckpt_path):
            pass
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
        
        # Set model properties and move to device
        model = model.to(device=device, dtype=dtype)
        
        return model, model_kwargs

    def __init__(
        self,
        input_size=1024,
        in_channels=4,
        hidden_size=1024,
        context_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_type='layer',
        qk_norm_type='rms',
        qk_norm=False,
        text_len=257,
        with_decoupled_ca=False,
        additional_cond_hidden_state=768,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        use_pos_emb=False,
        use_attention_pooling=True,
        guidance_cond_proj_dim=None,
        qkv_bias=True,
        num_moe_layers: int = 6,
        num_experts: int = 8,
        moe_top_k: int = 2,
        vggt_attn = False,
        vggt_text_states_dim = 0,
        vggt_feature_fusion_enabled = False,
        wan_style_timestep_modulation = False,
        time_freq_dim = 256,
        time_factor = 1000,
        pc_input = False,
        use_gradient_checkpointing=False,
        use_deepspeed_ckpt=False,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm if norm_type == 'layer' else nn.RMSNorm
        self.qk_norm = nn.RMSNorm if qk_norm_type == 'rms' else nn.LayerNorm
        self.context_dim = context_dim

        self.with_decoupled_ca = with_decoupled_ca
        self.decoupled_ca_dim = decoupled_ca_dim
        self.decoupled_ca_weight = decoupled_ca_weight
        self.use_pos_emb = use_pos_emb
        self.use_attention_pooling = use_attention_pooling
        self.guidance_cond_proj_dim = guidance_cond_proj_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_deepspeed_ckpt = use_deepspeed_ckpt

        self.text_len = text_len

        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size, hidden_size * 4, cond_proj_dim=guidance_cond_proj_dim)

        # Will use fixed sin-cos embedding:
        if self.use_pos_emb:
            self.register_buffer("pos_embed", torch.zeros(1, input_size, hidden_size))
            pos = np.arange(self.input_size, dtype=np.float32)
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], pos)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.use_attention_pooling = use_attention_pooling
        if use_attention_pooling:
            self.pooler = AttentionPool(self.text_len, context_dim, num_heads=8, output_dim=1024)
            self.extra_embedder = nn.Sequential(
                nn.Linear(1024, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, hidden_size, bias=True),
            )

        if with_decoupled_ca:
            self.additional_cond_hidden_state = additional_cond_hidden_state
            self.additional_cond_proj = nn.Sequential(
                nn.Linear(additional_cond_hidden_state, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, 1024, bias=True),
            )

        # HUnYuanDiT Blocks
        self.blocks = nn.ModuleList([
            HunYuanDiTBlock(hidden_size=hidden_size,
                            c_emb_size=hidden_size,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            text_states_dim=context_dim,
                            qk_norm=qk_norm,
                            norm_layer=self.norm,
                            qk_norm_layer=self.qk_norm,
                            skip_connection=layer > depth // 2,
                            with_decoupled_ca=with_decoupled_ca,
                            decoupled_ca_dim=decoupled_ca_dim,
                            decoupled_ca_weight=decoupled_ca_weight,
                            qkv_bias=qkv_bias,
                            use_moe=True if depth - layer <= num_moe_layers else False,
                            num_experts=num_experts,
                            moe_top_k=moe_top_k,
                            vggt_attn=vggt_attn,
                            vggt_text_states_dim=vggt_text_states_dim,
                            vggt_feature_fusion_enabled=vggt_feature_fusion_enabled,
                            wan_style_timestep_modulation=wan_style_timestep_modulation,
                            pc_input=pc_input,
                            )
            for layer in range(depth)
        ])
        self.depth = depth

        self.attn_vggt_mlp = None

        # use wan style timestep modulation
        self.wan_style_timestep_modulation = wan_style_timestep_modulation
        if self.wan_style_timestep_modulation:
            self.time_freq_dim = time_freq_dim
            self.time_factor = time_factor
            self.cross_attn_time_embedding = nn.Sequential(
                nn.Linear(time_freq_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
            self.cross_attn_time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 3))

        self.final_layer = FinalLayer(hidden_size, self.out_channels)


    def get_t_xt(self, x, t, **kwargs):
        t = self.t_embedder(t, condition=kwargs.get('guidance_cond'))
        x = self.x_embedder(x)

        if self.use_pos_emb:
            pos_embed = self.pos_embed.to(x.dtype)
            x = x + pos_embed

        if self.use_attention_pooling:
            raise NotImplementedError("Attention pooling is not implemented")
        else:
            c = t

        x = torch.cat([c, x], dim=1)

        return x, c, t

    def forward(self, x, t, contexts, **kwargs):


        #else:

        x, c, _ = self.get_t_xt(x, t, **kwargs)

        if self.wan_style_timestep_modulation:
            t = t.clamp(0, 1) * self.time_factor
            e = self.cross_attn_time_embedding(
                sinusoidal_embedding_1d(self.time_freq_dim, t).float())
            e = self.cross_attn_time_projection(e).unflatten(1, (3, self.hidden_size))
        else:
            e = None

        if 'self_attn_concat' in contexts:
            if self.use_pos_emb:
                raise NotImplementedError("Self-attention concatenation is not implemented with position embedding")
            cond_x = self.x_embedder(contexts['self_attn_concat'])
            keep_dim = x.shape[1]
            x = torch.cat([x, cond_x], dim=1)

        cond = contexts['main']

        if self.with_decoupled_ca:
            additional_cond = self.additional_cond_proj(contexts['additional'])
            cond = torch.cat([cond, additional_cond], dim=1)

        if self.attn_vggt_mlp is not None:
            vggt_text_states = self.attn_vggt_mlp(contexts['attn_vggt_tokens'])
        else:
            vggt_text_states = contexts.get('attn_vggt_tokens', None)

        def create_custom_forward(module):
            def custom_forward(* inputs):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): 
                    return module(* inputs)
            return custom_forward

        skip_value_list = []
        for layer, block in enumerate(self.blocks):
            skip_value = None if layer <= self.depth // 2 else skip_value_list.pop()
            if self.use_gradient_checkpointing:
                if self.use_deepspeed_ckpt:
                    x = deepspeed.checkpointing.non_reentrant_checkpoint(
                        create_custom_forward(block),
                        x, c, cond, skip_value, vggt_text_states, contexts.get('attn_vggt_tokens_pos', None), e,
                    )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        self.blocks[layer],
                        x, c, cond, skip_value, vggt_text_states, contexts.get('attn_vggt_tokens_pos', None), e,
                        use_reentrant=False,
                    )
            else:
                x = block(x, c, cond, skip_value=skip_value, 
                    vggt_text_states=vggt_text_states,
                    vggt_text_states_pos=contexts.get('attn_vggt_tokens_pos', None), e=e)
            if layer < self.depth // 2:
                skip_value_list.append(x)

        if 'self_attn_concat' in contexts:
            x = x[:, :keep_dim]

        x = self.final_layer(x)
        return x

def init_weights(net):
    """
    Initialize weights for decoder blocks. Works with any nn.Module or nn.ModuleList.
    
    Args:
        decoder_blocks: Any nn.Module, nn.ModuleList, or object with .modules() method
    """
    for module in net.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

        elif isinstance(module, nn.Conv2d):
            # If any Conv2d is used (not shown in your current code), initialize properly
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

class AdaptivePoolTokenCompressor(nn.Module):
    """
    Compress (B,S,C,H,W) -> (B,S,P,D) by pooling to a fixed Gh x Gw grid, then a 1x1 conv.
    Useful when you want an exact P = Gh*Gw regardless of H/W.

    Args:
        in_ch: input channels (C)
        embed_dim: token dim (D)
        grid_h: pooled grid height
        grid_w: pooled grid width
        add_ln: apply LayerNorm to token dim after projection
        mlp: if True, use a small MLP on tokens after projection
    """
    def __init__(self, in_ch: int = 128, embed_dim: int = 2048,
                 grid_h: int = 37, grid_w: int = 37, add_ln: bool = True, mlp: bool = False):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((grid_h, grid_w))
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=1, stride=1)
        self.ln = nn.LayerNorm(embed_dim) if add_ln else nn.Identity()
        if mlp:
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim),
            )
        else:
            self.mlp = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,S,C,H,W)
        returns: (B,S,P,D) with P = grid_h * grid_w
        """
        B, S, C, H, W = x.shape
        x = x.reshape(B * S, C, H, W)

        # pool to fixed grid, then project channels -> D
        x = self.pool(x)                  # (BS, C, Gh, Gw)
        x = self.proj(x)                  # (BS, D, Gh, Gw)

        BS, D, Gh, Gw = x.shape
        P = Gh * Gw
        x = x.permute(0, 2, 3, 1).reshape(B, S, P, D)  # (B,S,P,D)

        x = self.ln(x)
        if self.mlp is not None:
            # token MLP (pre-norm style)
            x = x + self.mlp(x)
        return x

class AdaptivePoolTokenCompressor_Omni(AdaptivePoolTokenCompressor):
    """
    Extended version of AdaptivePoolTokenCompressor with additional MLP processing
    and learnable conditioning signal.
    
    Args:
        in_ch: input channels (C)
        embed_dim: token dim (D)
        grid_h: pooled grid height
        grid_w: pooled grid width
        add_ln: apply LayerNorm to token dim after projection
        mlp: if True, use a small MLP on tokens after projection
        width: output dimension for the additional MLP and cond_signal
    """
    def __init__(self, in_ch: int = 128, embed_dim: int = 1024,
                 grid_h: int = 37, grid_w: int = 37, add_ln: bool = True, 
                 mlp: bool = False, width: int = 1024):
        super().__init__(in_ch, embed_dim, grid_h, grid_w, add_ln, mlp)
        
        # Additional two-layer MLP to process output to cond
        self.cond_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, width),
        )
        
        # Learnable conditioning signal parameters
        self.cond_signal_embedding = nn.Embedding(1, 8)
        self.cond_signal_linear = nn.Linear(8, width)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,S,C,H,W)
        returns: (B, S*P+10, width) where cond_signal (10 tokens) is concatenated to reshaped cond
        """
        # Get base output from parent class
        base_output = super().forward(x)  # (B,S,P,D)
        
        # Process through additional MLP to get cond
        cond = self.cond_mlp(base_output)  # (B,S,P,width)
        
        B, S, P, width = cond.shape
        
        # Reshape cond to (B, S*P, width)
        cond = cond.reshape(B, S * P, width)
        
        # Compute conditioning signal
        cond_signal = self.cond_signal_embedding(torch.tensor([0], device=cond.device))
        cond_signal = self.cond_signal_linear(cond_signal)  # (1, width)
        cond_signal = cond_signal.unsqueeze(0).repeat(B, 10, 1)  # (B, 10, width)
        
        # Concatenate along the token dimension (dim=1)
        output = torch.cat([cond, cond_signal], dim=1)  # (B, S*P+10, width)
        
        return output