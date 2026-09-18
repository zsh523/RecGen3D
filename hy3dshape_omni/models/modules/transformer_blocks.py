# -*- coding: utf-8 -*-
# Modified by the RecGen3D authors, 2026.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from hy3dshape_omni.models.modules.checkpoint import checkpoint


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
        qkv_bias: bool,
        flash: bool = False,
        use_checkpoint: bool = False,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadAttention(device=device, dtype=dtype, heads=heads, n_ctx=n_ctx, flash=flash,
                                               width=width, norm_layer=norm_layer, qk_norm=qk_norm)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), self.use_checkpoint)
        x = self.drop_path(self.c_proj(x))
        return x


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, heads: int, n_ctx: int, flash: bool = False,
                 width=None, qk_norm=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads
        self.n_ctx = n_ctx
        self.flash = flash
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)
        # import pdb; pdb.set_trace()

        if self.flash:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=True
            ):
                q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))
                out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bs, n_ctx, -1)
        else:
            weight = torch.einsum(
                "bthc,bshc->bhts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
        qkv_bias: bool = True,
        flash: bool = False,
        use_checkpoint: bool = False,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.attn = MultiheadAttention(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate
        )
        self.ln_1 = norm_layer(width, device=device, dtype=dtype, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale, drop_path_rate=drop_path_rate)
        self.ln_2 = norm_layer(width, device=device, dtype=dtype, elementwise_affine=True, eps=1e-6)

    def _forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward(self, x: torch.Tensor):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)


class DINOResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        context_dim: int = None,
        init_scale: float = 1.0,
        qkv_bias: bool = True,
        flash: bool = False,
        use_checkpoint: bool = False
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.attn = MultiheadAttention(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=use_checkpoint,
        )
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width, device=device, dtype=dtype)

        if context_dim is not None:
            self.ln_3 = nn.LayerNorm(width, device=device, dtype=dtype)
            self.cross_attn = MultiheadCrossAttention(
                device=device,
                dtype=dtype,
                width=width,
                heads=heads,
                data_width=context_dim,
                init_scale=init_scale,
                qkv_bias=qkv_bias,
                use_checkpoint=use_checkpoint,
            )

    def _forward(self, x: torch.Tensor, context: torch.Tensor = None, dino_context: Optional[torch.Tensor] = None):
        if dino_context is not None:
            residual = self.attn(self.ln_1(torch.cat([x, dino_context], dim=1)))[:, :x.shape[1], ...]
            x = x + residual
        else:
            x = x + self.attn(self.ln_1(x))
        if context is not None:
            x = x + self.cross_attn(self.ln_3(x), context)
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, dino_context: Optional[torch.Tensor] = None):
        return checkpoint(self._forward, (x, context, dino_context), self.parameters(), flag=self.use_checkpoint)


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        width: int,
        heads: int,
        init_scale: float,
        qkv_bias: bool = True,
        flash: bool = False,
        n_data: Optional[int] = None,
        data_width: Optional[int] = None,
        use_checkpoint: bool = False,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias, device=device, dtype=dtype)
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadCrossAttention(
            device=device, dtype=dtype, heads=heads, n_data=n_data, flash=flash, width=width, norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        init_linear(self.c_q, init_scale)
        init_linear(self.c_kv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x, data):
        x = self.c_q(x)
        data = self.c_kv(data)
        x = checkpoint(self.attention, (x, data), (), self.use_checkpoint)
        x = self.c_proj(x)
        return x


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, heads: int,
                 flash: bool = False, n_data: Optional[int] = None, width=None, qk_norm=False, norm_layer=nn.LayerNorm):

        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads
        self.n_data = n_data
        self.flash = flash
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)
        k = self.k_norm(k)
        q = self.q_norm(q)

        if self.flash:
            with torch.backends.cuda.sdp_kernel(enable_flash=True,
                enable_math=False,
                enable_mem_efficient=True
            ):
                q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))
                out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bs, n_ctx, -1)
        else:
            weight = torch.einsum(
                "bthc,bshc->bhts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        n_data: Optional[int] = None,
        width: int,
        heads: int,
        data_width: Optional[int] = None,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        flash: bool = False,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        expand_ratio=4,
    ):
        super().__init__()

        if data_width is None:
            data_width = width

        self.attn = MultiheadCrossAttention(
            device=device,
            dtype=dtype,
            n_data=n_data,
            width=width,
            heads=heads,
            data_width=data_width,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        self.ln_1 = norm_layer(width, device=device, dtype=dtype, elementwise_affine=True, eps=1e-6)
        self.ln_2 = norm_layer(data_width, device=device, dtype=dtype, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale, expand_ratio=expand_ratio)
        self.ln_3 = norm_layer(width, device=device, dtype=dtype, elementwise_affine=True, eps=1e-6)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x


class MLP(nn.Module):
    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 width: int,
                 init_scale: float,
                 output_width: int = None,
                 drop_path_rate: float = 0.0,
                 expand_ratio: int = 4):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * expand_ratio, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * expand_ratio, 
                                output_width if output_width is not None else width, 
                                device=device,
                                dtype=dtype)
        self.gelu = nn.GELU()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.drop_path(self.c_proj(self.gelu(self.c_fc(x))))


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        flash: bool = False,
        use_checkpoint: bool = False,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                    qkv_bias=qkv_bias,
                    flash=flash,
                    use_checkpoint=use_checkpoint,
                    norm_layer=norm_layer,
                    qk_norm=qk_norm,
                    drop_path_rate=drop_path_rate
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x
