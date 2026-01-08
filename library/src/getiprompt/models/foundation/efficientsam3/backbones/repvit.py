# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RepViT backbone for EfficientSAM3.

Adapted from: https://github.com/SimonZeng7108/efficientsam3
"""

import warnings

import torch
import torch.nn as nn
from timm.layers.squeeze_excite import SqueezeExcite
from timm.layers.weight_init import trunc_normal_

__all__ = [
    "RepViT",
    "repvit_m0_9",
    "repvit_m1_1",
    "repvit_m2_3",
]


def _make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
    """Make value divisible by divisor.

    This function ensures that all layers have a channel number that is divisible by 8.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


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
        resolution: int = -10000,
    ) -> None:
        super().__init__()
        self.add_module("c", nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module("bn", nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

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
            device=c.weight.device,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(nn.Module):
    """Residual connection with optional dropout."""

    def __init__(self, m: nn.Module, drop: float = 0.0) -> None:
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(
                1 - self.drop
            ).detach()
        return x + self.m(x)

    @torch.no_grad()
    def fuse(self) -> nn.Module:
        """Fuse residual connection for inference."""
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert m.groups == m.in_channels
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, nn.Conv2d):
            m = self.m
            assert m.groups != m.in_channels
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        return self


class RepVGGDW(nn.Module):
    """RepVGG Depthwise block."""

    def __init__(self, ed: int) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = nn.BatchNorm2d(ed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        """Fuse RepVGGDW block for inference."""
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = nn.functional.pad(conv1_w, [1, 1, 1, 1])
        identity = nn.functional.pad(
            torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
            [1, 1, 1, 1],
        )

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class RepViTBlock(nn.Module):
    """RepViT building block."""

    def __init__(
        self,
        inp: int,
        hidden_dim: int,
        oup: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        use_hs: bool,
    ) -> None:
        super().__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert hidden_dim == 2 * inp

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                )
            )
        else:
            assert self.identity
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_mixer(self.token_mixer(x))


class BN_Linear(nn.Sequential):
    """BatchNorm followed by Linear layer."""

    def __init__(self, a: int, b: int, bias: bool = True, std: float = 0.02) -> None:
        super().__init__()
        self.add_module("bn", nn.BatchNorm1d(a))
        self.add_module("l", nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self) -> nn.Linear:
        """Fuse BatchNorm and Linear for inference."""
        bn, linear = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = linear.weight * w[None, :]
        if linear.bias is None:
            b = b @ linear.weight.T
        else:
            b = (linear.weight @ b[:, None]).view(-1) + linear.bias
        m = nn.Linear(w.size(1), w.size(0), device=linear.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Classifier(nn.Module):
    """Classification head with optional distillation."""

    def __init__(self, dim: int, num_classes: int, distillation: bool = True) -> None:
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.distillation = distillation
        if distillation:
            self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x

    @torch.no_grad()
    def fuse(self) -> nn.Linear:
        """Fuse classifiers for inference."""
        classifier = self.classifier.fuse()
        if self.distillation:
            classifier_dist = self.classifier_dist.fuse()
            classifier.weight += classifier_dist.weight
            classifier.bias += classifier_dist.bias
            classifier.weight /= 2
            classifier.bias /= 2
        return classifier


class RepViT(nn.Module):
    """RepViT model.

    A re-parameterizable vision transformer for efficient inference.
    """

    def __init__(
        self,
        cfgs: list[list],
        num_classes: int = 1000,
        distillation: bool = False,
    ) -> None:
        super().__init__()
        self.cfgs = cfgs

        input_channel = self.cfgs[0][2]
        patch_embed = nn.Sequential(
            Conv2d_BN(3, input_channel // 2, 3, 2, 1),
            nn.GELU(),
            Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1),
        )
        layers = [patch_embed]

        block = RepViTBlock
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel

        self.features = nn.ModuleList(layers)
        self.classifier = Classifier(output_channel, num_classes, distillation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for f in self.features:
            x = f(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.classifier(x)
        return x


# Suppress warning about overwriting repvit models in registry
warnings.filterwarnings("ignore", message="Overwriting .* in registry")


def repvit_m0_9(pretrained: bool = False, num_classes: int = 1000, distillation: bool = False) -> RepViT:
    """Create RepViT-M0.9 model."""
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 2, 48, 1, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 96, 0, 0, 2],
        [3, 2, 96, 1, 0, 1],
        [3, 2, 96, 0, 0, 1],
        [3, 2, 96, 0, 0, 1],
        [3, 2, 192, 0, 1, 2],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 384, 0, 1, 2],
        [3, 2, 384, 1, 1, 1],
        [3, 2, 384, 0, 1, 1],
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)


def repvit_m1_1(pretrained: bool = False, num_classes: int = 1000, distillation: bool = False) -> RepViT:
    """Create RepViT-M1.1 model."""
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 128, 0, 0, 2],
        [3, 2, 128, 1, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 256, 0, 1, 2],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 512, 0, 1, 2],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 0, 1, 1],
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)


def repvit_m2_3(pretrained: bool = False, num_classes: int = 1000, distillation: bool = False) -> RepViT:
    """Create RepViT-M2.3 model."""
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 2, 80, 1, 0, 1],
        [3, 2, 80, 0, 0, 1],
        [3, 2, 80, 1, 0, 1],
        [3, 2, 80, 0, 0, 1],
        [3, 2, 80, 1, 0, 1],
        [3, 2, 80, 0, 0, 1],
        [3, 2, 80, 0, 0, 1],
        [3, 2, 160, 0, 0, 2],
        [3, 2, 160, 1, 0, 1],
        [3, 2, 160, 0, 0, 1],
        [3, 2, 160, 1, 0, 1],
        [3, 2, 160, 0, 0, 1],
        [3, 2, 160, 1, 0, 1],
        [3, 2, 160, 0, 0, 1],
        [3, 2, 160, 0, 0, 1],
        [3, 2, 320, 0, 1, 2],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 640, 0, 1, 2],
        [3, 2, 640, 1, 1, 1],
        [3, 2, 640, 0, 1, 1],
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)
