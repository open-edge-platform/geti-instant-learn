# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Morphological post-processors for mask cleaning.

Uses pure PyTorch operations (``max_pool2d`` trick) so they are
fully ONNX-exportable. No OpenCV/SciPy dependency required.

Morphological opening removes small protrusions and isolated pixels.
Morphological closing fills small holes inside masks.
"""

from __future__ import annotations

import torch
from torch.nn import functional

from instantlearn.components.postprocessing.base import PostProcessor


def _erode(masks: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Erode binary masks using min-pooling (= negated max-pool of negated input).

    Args:
        masks: Float masks ``[N, H, W]`` with values in {0, 1}.
        kernel_size: Size of the square erosion kernel.

    Returns:
        Eroded masks ``[N, H, W]``.
    """
    padding = kernel_size // 2
    # min-pool = negate → max-pool → negate
    x = masks.unsqueeze(1)  # [N, 1, H, W]
    x = -functional.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=padding)
    return x.squeeze(1)


def _dilate(masks: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Dilate binary masks using max-pooling.

    Args:
        masks: Float masks ``[N, H, W]`` with values in {0, 1}.
        kernel_size: Size of the square dilation kernel.

    Returns:
        Dilated masks ``[N, H, W]``.
    """
    padding = kernel_size // 2
    x = masks.unsqueeze(1)  # [N, 1, H, W]
    x = functional.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
    return x.squeeze(1)


class MorphologicalOpening(PostProcessor):
    """Morphological opening: erode then dilate.

    Removes small foreground noise (isolated pixels, thin protrusions)
    while preserving the overall shape of larger regions.

    Uses ``max_pool2d`` for ONNX-traceable dilation/erosion.

    Args:
        kernel_size: Size of the square structuring element.
            Must be odd. Default: ``3``.
    """

    def __init__(self, kernel_size: int = 3) -> None:
        """Initialize with the structuring element size.

        Raises:
            ValueError: If kernel_size is even.
        """
        super().__init__()
        if kernel_size % 2 == 0:
            msg = f"kernel_size must be odd, got {kernel_size}"
            raise ValueError(msg)
        self.kernel_size = kernel_size

    def forward(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply morphological opening to each mask.

        Args:
            masks: Binary masks ``[N, H, W]``.
            scores: Confidence scores ``[N]``.
            labels: Category labels ``[N]``.

        Returns:
            Cleaned (masks, scores, labels). Mask count is unchanged.
        """
        if not torch.onnx.is_in_onnx_export() and masks.size(0) == 0:
            return masks, scores, labels

        float_masks = masks.float()
        opened = _dilate(_erode(float_masks, self.kernel_size), self.kernel_size)
        return opened > 0.5, scores, labels


class MorphologicalClosing(PostProcessor):
    """Morphological closing: dilate then erode.

    Fills small holes and gaps inside masks while preserving the
    overall outline.

    Uses ``max_pool2d`` for ONNX-traceable dilation/erosion.

    Args:
        kernel_size: Size of the square structuring element.
            Must be odd. Default: ``3``.
    """

    def __init__(self, kernel_size: int = 3) -> None:
        """Initialize with the structuring element size.

        Raises:
            ValueError: If kernel_size is even.
        """
        super().__init__()
        if kernel_size % 2 == 0:
            msg = f"kernel_size must be odd, got {kernel_size}"
            raise ValueError(msg)
        self.kernel_size = kernel_size

    def forward(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply morphological closing to each mask.

        Args:
            masks: Binary masks ``[N, H, W]``.
            scores: Confidence scores ``[N]``.
            labels: Category labels ``[N]``.

        Returns:
            Cleaned (masks, scores, labels). Mask count is unchanged.
        """
        if not torch.onnx.is_in_onnx_export() and masks.size(0) == 0:
            return masks, scores, labels

        float_masks = masks.float()
        closed = _erode(_dilate(float_masks, self.kernel_size), self.kernel_size)
        return closed > 0.5, scores, labels
