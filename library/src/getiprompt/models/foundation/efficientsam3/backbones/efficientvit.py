# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientViT backbone for EfficientSAM3.

Adapted from: https://github.com/SimonZeng7108/efficientsam3
Original: https://github.com/mit-han-lab/efficientvit
"""

import warnings

import torch
from timm.layers.drop import DropPath as TimmDropPath
from timm.layers.helpers import to_2tuple
from timm.layers.weight_init import trunc_normal_
from torch import nn

__all__ = [
    "EfficientViT",
    "efficientvit_b0",
    "efficientvit_b1",
    "efficientvit_b2",
]

# Suppress warning about overwriting models in registry
warnings.filterwarnings("ignore", message="Overwriting .* in registry")


class Conv2d_BN(nn.Sequential):
    """Conv2d with BatchNorm."""

    def __init__(
        self,
        a: int,
        b: int,
        ks: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bn_weight_init: float = 1,
    ) -> None:
        super().__init__()
        self.add_module("c", nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = nn.BatchNorm2d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        """Fuse Conv2d and BatchNorm for inference."""
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class DropPath(TimmDropPath):
    """DropPath with drop_prob attribute."""

    def __init__(self, drop_prob: float | None = None) -> None:
        super().__init__(drop_prob=drop_prob)
        self.drop_prob = drop_prob

    def __repr__(self) -> str:
        msg = super().__repr__()
        return f"{msg}(drop_prob={self.drop_prob})"


class DSConv(nn.Module):
    """Depthwise separable convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.depth_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=use_bias,
        )
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class LiteMLA(nn.Module):
    """Lightweight Multi-head Linear Attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.head_dim = out_channels // heads

        self.qkv = Conv2d_BN(in_channels, out_channels * 3, 1)
        self.proj = Conv2d_BN(out_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B, C, H, W = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape to (B, heads, head_dim, H*W)
        q = q.reshape(B, self.heads, self.head_dim, H * W)
        k = k.reshape(B, self.heads, self.head_dim, H * W)
        v = v.reshape(B, self.heads, self.head_dim, H * W)

        # Normalize
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        # Attention: (B, heads, head_dim, head_dim) @ (B, heads, head_dim, H*W)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhdn,bhde->bhen", q, context)

        # Reshape back
        out = out.reshape(B, -1, H, W)
        out = self.proj(out)

        return out


class EfficientViTBlock(nn.Module):
    """EfficientViT block with LiteMLA."""

    def __init__(
        self,
        in_channels: int,
        heads: int,
        expand_ratio: float = 4.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        self.context_module = nn.Sequential(
            Conv2d_BN(in_channels, in_channels, 1),
            nn.GELU(),
            LiteMLA(in_channels, in_channels, heads=heads),
        )

        hidden_dim = int(in_channels * expand_ratio)
        self.local_module = nn.Sequential(
            Conv2d_BN(in_channels, hidden_dim, 1),
            nn.GELU(),
            DSConv(hidden_dim, hidden_dim, kernel_size=3),
            nn.GELU(),
            Conv2d_BN(hidden_dim, in_channels, 1),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x + self.drop_path(self.context_module(x))
        x = x + self.drop_path(self.local_module(x))
        return x


class PatchEmbed(nn.Module):
    """Patch embedding layer."""

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        resolution: int = 224,
    ) -> None:
        super().__init__()
        img_size = to_2tuple(resolution)
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        # Stem: overlapping convolution for better low-level features
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, embed_dim // 2, kernel_size=3, stride=2, pad=1),
            nn.GELU(),
            Conv2d_BN(embed_dim // 2, embed_dim, kernel_size=3, stride=2, pad=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.seq(x)


class EfficientViTStage(nn.Module):
    """EfficientViT stage with multiple blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        heads: int,
        expand_ratio: float = 4.0,
        drop_path: list[float] | None = None,
        downsample: bool = True,
    ) -> None:
        super().__init__()

        if drop_path is None:
            drop_path = [0.0] * depth

        # Downsample if needed
        if downsample:
            self.downsample = nn.Sequential(
                Conv2d_BN(in_channels, out_channels, kernel_size=2, stride=2),
                nn.GELU(),
            )
        else:
            self.downsample = Conv2d_BN(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # Stack of EfficientViT blocks
        self.blocks = nn.ModuleList(
            [
                EfficientViTBlock(
                    in_channels=out_channels,
                    heads=heads,
                    expand_ratio=expand_ratio,
                    drop_path=drop_path[i],
                )
                for i in range(depth)
            ],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class EfficientViT(nn.Module):
    """EfficientViT backbone for EfficientSAM3.

    Args:
        in_chans: Number of input channels
        img_size: Input image size
        embed_dim: Initial embedding dimension
        depths: Number of blocks in each stage
        num_heads: Number of attention heads in each stage
        expand_ratio: Expansion ratio for MLP
        drop_path_rate: Stochastic depth rate
    """

    def __init__(
        self,
        in_chans: int = 3,
        img_size: int = 224,
        embed_dim: list[int] = [32, 64, 128],
        depths: list[int] = [1, 2, 2],
        num_heads: list[int] = [4, 4, 4],
        expand_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_stages = len(depths)
        self.num_features = embed_dim[-1]

        # Patch embedding (stem)
        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            resolution=img_size,
        )

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build stages
        self.stages = nn.ModuleList()
        in_dim = embed_dim[0]

        for i in range(self.num_stages):
            stage = EfficientViTStage(
                in_channels=in_dim,
                out_channels=embed_dim[i],
                depth=depths[i],
                heads=num_heads[i],
                expand_ratio=expand_ratio,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                downsample=(i > 0),  # Downsample for stages after the first
            )
            self.stages.append(stage)
            in_dim = embed_dim[i]

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, (2.0 / fan_out) ** 0.5)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output features from the last stage
        """
        x = self.patch_embed(x)

        for stage in self.stages:
            x = stage(x)

        return x


def efficientvit_b0(img_size: int = 224, **kwargs) -> EfficientViT:
    """EfficientViT-B0 model (smallest variant, ~0.68M params)."""
    return EfficientViT(
        img_size=img_size,
        embed_dim=[16, 32, 64],
        depths=[1, 2, 2],
        num_heads=[2, 2, 4],
        expand_ratio=4.0,
        **kwargs,
    )


def efficientvit_b1(img_size: int = 224, **kwargs) -> EfficientViT:
    """EfficientViT-B1 model (medium variant, ~4.64M params)."""
    return EfficientViT(
        img_size=img_size,
        embed_dim=[32, 64, 128],
        depths=[1, 2, 3],
        num_heads=[4, 4, 4],
        expand_ratio=4.0,
        **kwargs,
    )


def efficientvit_b2(img_size: int = 224, **kwargs) -> EfficientViT:
    """EfficientViT-B2 model (largest variant, ~14.98M params)."""
    return EfficientViT(
        img_size=img_size,
        embed_dim=[48, 96, 192],
        depths=[2, 3, 4],
        num_heads=[4, 4, 8],
        expand_ratio=4.0,
        **kwargs,
    )
