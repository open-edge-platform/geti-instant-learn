# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Common modules for Per SAM model."""

import torch
from torch import nn


class MLPBlock(nn.Module):
    """MLPBlock."""

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: type[nn.Module] = nn.GELU,
    ) -> None:
        """MLPBlock.

        Args:
            embedding_dim (int): Embedding dimension.
            mlp_dim (int): MLP dimension.
            act (type[nn.Module]): Activation function.
        """
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    """LayerNorm2d."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        """LayerNorm2d.

        Args:
            num_channels (int): Number of channels.
            eps (float): Epsilon.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]
