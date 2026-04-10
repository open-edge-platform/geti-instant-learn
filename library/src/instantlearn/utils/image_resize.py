# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for aspect-ratio-preserving image and mask resizing."""

from dataclasses import dataclass

import torch
from torch.nn import functional


@dataclass
class DownscaleResult:
    """Result of downscaling an image, with metadata needed to reverse the operation.

    Attributes:
        image: The (possibly) downscaled image [C, H', W'].
        scale: Scale factor applied (1.0 if no downscaling needed).
        orig_size: Original (H, W) before downscaling.
    """

    image: torch.Tensor
    scale: float
    orig_size: tuple[int, int]

    @property
    def is_scaled(self) -> bool:
        """Whether the image was actually downscaled."""
        return self.scale < 1.0


def downscale_image(image: torch.Tensor, max_side: int) -> DownscaleResult:
    """Downscale an image so its longest side does not exceed max_side.

    Preserves aspect ratio. Returns the image unchanged if already small enough.

    Args:
        image: Image tensor [C, H, W] (any dtype).
        max_side: Maximum allowed side length.

    Returns:
        DownscaleResult with the image, scale factor, and original size.
    """
    h, w = image.size(1), image.size(2)
    longest = max(h, w)

    if longest <= max_side:
        return DownscaleResult(image=image, scale=1.0, orig_size=(h, w))

    scale = max_side / longest
    new_h = int(h * scale)
    new_w = int(w * scale)
    resized = (
        functional.interpolate(
            image.unsqueeze(0).float(),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .to(image.dtype)
    )
    return DownscaleResult(image=resized, scale=scale, orig_size=(h, w))


def upscale_masks(masks: torch.Tensor, orig_size: tuple[int, int]) -> torch.Tensor:
    """Upscale boolean masks back to their original resolution.

    Uses nearest-neighbor interpolation to preserve sharp mask boundaries.

    Args:
        masks: Boolean masks [N, H', W'] at reduced resolution.
        orig_size: Target (H, W) to restore.

    Returns:
        Masks [N, H, W] at original resolution.
    """
    if masks.numel() == 0:
        return masks
    return (
        functional.interpolate(
            masks.unsqueeze(1).float(),
            size=orig_size,
            mode="nearest",
        )
        .squeeze(1)
        .bool()
    )
