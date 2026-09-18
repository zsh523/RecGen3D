# -*- coding: utf-8 -*-
# Modified by the RecGen3D authors, 2026.

import math
import torch
import torch.nn as nn
from typing import Optional

from hy3dshape_omni.models.modules.checkpoint import checkpoint
from hy3dshape_omni.models.modules.transformer_blocks import (
    init_linear,
    MLP,
    MultiheadCrossAttention,
    MultiheadAttention,
    ResidualAttentionBlock,
    DINOResidualAttentionBlock
)


class AdaLayerNorm(nn.Module):
    def __init__(self,
                 device: torch.device,
                 dtype: torch.dtype,
                 width: int):
        super().__init__()

        self.silu = nn.SiLU(inplace=True)
        self.linear = nn.Linear(width, width * 2, device=device, dtype=dtype)
        self.layernorm = nn.LayerNorm(width, elementwise_affine=False, device=device, dtype=dtype)

    def forward(self, x, timestep):
        emb = self.linear(timestep)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class DitBlock(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        context_dim: int,
        qkv_bias: bool = False,
        init_scale: float = 1.0,
        flash: bool = False,
        use_checkpoint: bool = False,
        qk_norm: bool = False
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
            qk_norm=qk_norm,
        )
        self.ln_1 = AdaLayerNorm(device, dtype, width)

        if context_dim is not None:
            self.ln_2 = AdaLayerNorm(device, dtype, width)
            self.cross_attn = MultiheadCrossAttention(
                device=device,
                dtype=dtype,
                width=width,
                heads=heads,
                data_width=context_dim,
                init_scale=init_scale,
                qkv_bias=qkv_bias,
                flash=flash,
                qk_norm=qk_norm,
            )

        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_3 = AdaLayerNorm(device, dtype, width)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None):
        return checkpoint(self._forward, (x, t, context), self.parameters(), self.use_checkpoint)

    def _forward(self, x: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None):
        x = x + self.attn(self.ln_1(x, t))
        if context is not None:
            x = x + self.cross_attn(self.ln_2(x, t), context)
        x = x + self.mlp(self.ln_3(x, t))
        return x


class DiT(nn.Module):
    def __init__(
        self,
        *,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        context_dim: int,
        init_scale: float = 0.25,
        qkv_bias: bool = False,
        flash: bool = False,
        use_checkpoint: bool = False,
        qk_norm: bool = False
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers

        self.resblocks = nn.ModuleList(
            [
                DitBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    context_dim=context_dim,
                    qkv_bias=qkv_bias,
                    init_scale=init_scale,
                    flash=flash,
                    use_checkpoint=use_checkpoint,
                    qk_norm=qk_norm
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None):
        for block in self.resblocks:
            x = block(x, t, context)
        return x


class DINODitBlock(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        context_dim: int,
        qkv_bias: bool = False,
        init_scale: float = 1.0,
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
            flash=flash
        )
        self.ln_1 = AdaLayerNorm(device, dtype, width)

        if context_dim is not None:
            self.ln_2 = AdaLayerNorm(device, dtype, width)
            self.cross_attn = MultiheadCrossAttention(
                device=device,
                dtype=dtype,
                width=width,
                heads=heads,
                data_width=context_dim,
                init_scale=init_scale,
                qkv_bias=qkv_bias
            )

        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_3 = AdaLayerNorm(device, dtype, width)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: Optional[dict] = None,
                dino_context: Optional[torch.Tensor] = None):
        if dino_context is None:
            return checkpoint(self._forward, (x, t, context), self.parameters(), self.use_checkpoint)

        return checkpoint(self._forward, (x, t, context, dino_context), self.parameters(), self.use_checkpoint)

    def _forward(self, x: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None,
                 dino_context: Optional[torch.Tensor] = None):
        if dino_context is not None:
            residual = self.attn(self.ln_1(torch.cat([x, dino_context], dim=1), t))[:, :x.shape[1], ...]
            x = x + residual
        else:
            x = x + self.attn(self.ln_1(x, t))
        if context is not None:
            x = x + self.cross_attn(self.ln_2(x, t), context)
        x = x + self.mlp(self.ln_3(x, t))
        return x


class DINODiT(nn.Module):
    def __init__(
        self,
        *,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        context_dim: int,
        init_scale: float = 0.25,
        qkv_bias: bool = False,
        flash: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers

        self.resblocks = nn.ModuleList(
            [
                DINODitBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    context_dim=context_dim,
                    qkv_bias=qkv_bias,
                    init_scale=init_scale,
                    flash=flash,
                    use_checkpoint=use_checkpoint
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None,
                dino_context: Optional[torch.Tensor] = None):
        for block in self.resblocks:
            x = block(x, t, context, dino_context)
        return x


class DINODiTSkip(nn.Module):
    def __init__(
        self,
        *,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        context_dim: int,
        init_scale: float = 0.25,
        qkv_bias: bool = False,
        flash: bool = False,
        use_checkpoint: bool = False,
        skip_ln: bool = True,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers

        self.encoder = nn.ModuleList()
        for _ in range(layers // 2):
            resblock = DINODitBlock(
                device=device,
                dtype=dtype,
                n_ctx=n_ctx,
                width=width,
                heads=heads,
                context_dim=context_dim,
                qkv_bias=qkv_bias,
                init_scale=init_scale,
                flash=flash,
                use_checkpoint=use_checkpoint
            )
            self.encoder.append(resblock)

        self.decoder = nn.ModuleList()
        for _ in range(layers // 2):
            resblock = DINODitBlock(
                device=device,
                dtype=dtype,
                n_ctx=n_ctx,
                width=width,
                heads=heads,
                context_dim=context_dim,
                qkv_bias=qkv_bias,
                init_scale=init_scale,
                flash=flash,
                use_checkpoint=use_checkpoint
            )
            linear = nn.Linear(width * 2, width, device=device, dtype=dtype)
            init_linear(linear, init_scale)

            layer_norm = nn.LayerNorm(width, device=device, dtype=dtype) if skip_ln else None

            self.decoder.append(nn.ModuleList([resblock, linear, layer_norm]))

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None,
                dino_context: Optional[torch.Tensor] = None):
        enc_outputs = []
        for block in self.encoder:
            x = block(x, t, context, dino_context)
            enc_outputs.append(x)

        for i, (resblock, linear, layer_norm) in enumerate(self.decoder):
            x = torch.cat([enc_outputs.pop(), x], dim=-1)
            x = linear(x)

            if layer_norm is not None:
                x = layer_norm(x)

            x = resblock(x, t, context, dino_context)

        return x


class UNetDiffusionTransformer(nn.Module):
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
        qkv_bias: bool = False,
        skip_ln: bool = False,
        flash: bool = False,
        use_checkpoint: bool = False
    ):
        super().__init__()

        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers

        self.encoder = nn.ModuleList()
        for _ in range(layers):
            resblock = ResidualAttentionBlock(
                device=device,
                dtype=dtype,
                n_ctx=n_ctx,
                width=width,
                heads=heads,
                init_scale=init_scale,
                qkv_bias=qkv_bias,
                flash=flash,
                use_checkpoint=use_checkpoint
            )
            self.encoder.append(resblock)

        self.middle_block = ResidualAttentionBlock(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=use_checkpoint
        )

        self.decoder = nn.ModuleList()
        for _ in range(layers):
            resblock = ResidualAttentionBlock(
                device=device,
                dtype=dtype,
                n_ctx=n_ctx,
                width=width,
                heads=heads,
                init_scale=init_scale,
                qkv_bias=qkv_bias,
                flash=flash,
                use_checkpoint=use_checkpoint
            )
            linear = nn.Linear(width * 2, width, device=device, dtype=dtype)
            init_linear(linear, init_scale)

            layer_norm = nn.LayerNorm(width, device=device, dtype=dtype) if skip_ln else None

            self.decoder.append(nn.ModuleList([resblock, linear, layer_norm]))

    def forward(self, x: torch.Tensor):

        enc_outputs = []
        for block in self.encoder:
            x = block(x)
            enc_outputs.append(x)

        x = self.middle_block(x)

        for i, (resblock, linear, layer_norm) in enumerate(self.decoder):
            x = torch.cat([enc_outputs.pop(), x], dim=-1)
            x = linear(x)

            if layer_norm is not None:
                x = layer_norm(x)

            x = resblock(x)

        return x


class DINOUNetDiffusionTransformerCross(nn.Module):
    def __init__(
        self,
        *,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        context_dim: int = None,
        init_scale: float = 0.25,
        qkv_bias: bool = False,
        skip_ln: bool = False,
        flash: bool = False,
        use_checkpoint: bool = False
    ):
        super().__init__()

        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers

        self.encoder = nn.ModuleList()
        for _ in range(layers):
            resblock = DINOResidualAttentionBlock(
                device=device,
                dtype=dtype,
                n_ctx=n_ctx,
                width=width,
                heads=heads,
                init_scale=init_scale,
                qkv_bias=qkv_bias,
                flash=flash,
                use_checkpoint=use_checkpoint,
                context_dim=context_dim,
            )
            self.encoder.append(resblock)

        self.middle_block = DINOResidualAttentionBlock(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=use_checkpoint,
            context_dim=context_dim,
        )

        self.decoder = nn.ModuleList()
        for _ in range(layers):
            resblock = DINOResidualAttentionBlock(
                device=device,
                dtype=dtype,
                n_ctx=n_ctx,
                width=width,
                heads=heads,
                init_scale=init_scale,
                qkv_bias=qkv_bias,
                flash=flash,
                use_checkpoint=use_checkpoint,
                context_dim=context_dim,
            )
            linear = nn.Linear(width * 2, width, device=device, dtype=dtype)
            init_linear(linear, init_scale)

            layer_norm = nn.LayerNorm(width, device=device, dtype=dtype) if skip_ln else None

            self.decoder.append(nn.ModuleList([resblock, linear, layer_norm]))

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, dino_context: Optional[torch.Tensor] = None):

        enc_outputs = []
        for block in self.encoder:
            x = block(x, context, dino_context)
            enc_outputs.append(x)

        x = self.middle_block(x, context, dino_context)

        for i, (resblock, linear, layer_norm) in enumerate(self.decoder):
            x = torch.cat([enc_outputs.pop(), x], dim=-1)
            x = linear(x)

            if layer_norm is not None:
                x = layer_norm(x)

            x = resblock(x, context, dino_context)

        return x
