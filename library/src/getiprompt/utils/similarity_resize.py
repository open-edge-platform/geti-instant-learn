# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for resizing similarity maps."""

import math

import torch
from torch.nn import functional


def resize_similarity_map(
    similarities: torch.Tensor,
    target_size: tuple[int, int] | int,
    unpadded_image_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Resize similarity maps to target image size while removing padding.

    This function converts flat similarity tensors to square spatial grids,
    removes any padding (e.g., from SAM models), and resizes to the target dimensions.

    Args:
        similarities: Similarity tensor of shape (batch, num_features) or (batch, 1, H, W).
            If 2D with shape (batch, N), it will be reshaped to square spatial dimensions
            where N = H * W and H = W = sqrt(N).
        target_size: Target image size as (height, width) tuple or single int for square.
        unpadded_image_size: Original image size before padding as (height, width).
            If provided, padding will be removed before final resize.

    Returns:
        Resized similarity tensor of shape (batch, height, width) or (height, width)
        if batch dimension is 1.

    Examples:
        >>> import torch
        >>> # Flat similarities from 64x64 feature map
        >>> similarities = torch.randn(1, 4096)  # 64*64=4096
        >>> resized = resize_similarity_map(similarities, target_size=(256, 256))
        >>> resized.shape
        torch.Size([256, 256])
        >>>
        >>> # With padding removal (e.g., SAM models)
        >>> resized = resize_similarity_map(
        ...     similarities,
        ...     target_size=(256, 256),
        ...     unpadded_image_size=(240, 240)
        ... )
        >>> resized.shape
        torch.Size([256, 256])
        >>>
        >>> # Batch processing
        >>> similarities_batch = torch.randn(4, 1024)  # batch=4, 32x32
        >>> resized_batch = resize_similarity_map(similarities_batch, target_size=(128, 128))
        >>> resized_batch.shape
        torch.Size([4, 128, 128])
    """
    square_size = int(math.sqrt(similarities.shape[-1]))

    # Put in batched square shape
    similarities = similarities.reshape(
        similarities.shape[0],
        1,
        square_size,
        square_size,
    )

    # SAM models can in some cases add padding to the image, we need to remove it
    if unpadded_image_size is not None:
        similarities = functional.interpolate(
            similarities,
            size=max(unpadded_image_size),
            mode="bilinear",
            align_corners=False,
        )
        similarities = similarities[
            ...,
            : unpadded_image_size[0],
            : unpadded_image_size[1],
        ]

    # Resize to (original) target size
    similarities = functional.interpolate(
        similarities,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    # Squeeze batch dimension if batch size is 1
    if similarities.shape[0] == 1:
        similarities = similarities.squeeze(0)

    return similarities
