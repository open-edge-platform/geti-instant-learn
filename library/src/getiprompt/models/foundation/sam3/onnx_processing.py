# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ONNX-traceable image preprocessor for SAM3 model.

This module provides an ONNX-compatible image preprocessing pipeline that replaces
the transformers-dependent ImageProcessorFast. All operations are pure PyTorch to
ensure ONNX traceability.
"""

import torch
from torch import nn
from torch.nn import functional


class Sam3Preprocessor(nn.Module):
    """ONNX-traceable image preprocessor for SAM3.

    This preprocessor handles image resizing, padding, and normalization using
    only PyTorch operations, making it fully ONNX-traceable. It replaces the
    transformers-dependent ImageProcessorFast.

    Args:
        target_size: The target size for the longest dimension of the image.
                    Default: 1008 (standard SAM3 input size).

    Attributes:
        target_size: The target size for the longest dimension.
        mean: ImageNet normalization mean (registered as buffer).
        std: ImageNet normalization standard deviation (registered as buffer).

    Example:
        >>> import torch
        >>> preprocessor = Sam3Preprocessor(target_size=1008)
        >>> image = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8)
        >>> pixel_values, original_sizes = preprocessor(image)
        >>> pixel_values.shape
        torch.Size([1, 3, 1008, 1008])
    """

    def __init__(self, target_size: int = 1008) -> None:
        """Initialize the preprocessor.

        Args:
            target_size: The target size for the longest dimension. Default: 1008.
        """
        super().__init__()
        self.target_size = target_size

        # Register ImageNet normalization constants as buffers for ONNX compatibility
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape(1, 3, 1, 1),
        )

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> tuple[int, int]:
        """Compute the output size given input size and target long side length.

        Scales the image such that the longest dimension becomes long_side_length
        while maintaining aspect ratio.

        Args:
            oldh: Original image height.
            oldw: Original image width.
            long_side_length: The target length for the longest dimension.

        Returns:
            Tuple of (new_height, new_width) maintaining aspect ratio.

        Example:
            >>> Sam3Preprocessor.get_preprocess_shape(480, 640, 1008)
            (756, 1008)
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh = int(oldh * scale + 0.5)
        neww = int(oldw * scale + 0.5)
        return (newh, neww)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocess image for SAM3 inference.

        Handles input format conversion, resizing to target size, padding, and
        ImageNet normalization.

        Args:
            pixel_values: Input image tensor with shape (B, C, H, W).
                         Can be uint8 (0-255) or float (0-1).

        Returns:
            Tuple containing:
                - pixel_values: Preprocessed image tensor with shape (B, 3, target_size, target_size)
                  and ImageNet normalized values.
                - original_sizes: Tensor with shape (B, 2) containing [height, width] of input images.

        Example:
            >>> preprocessor = Sam3Preprocessor(target_size=1008)
            >>> # uint8 input
            >>> img_uint8 = torch.randint(0, 256, (2, 3, 480, 640), dtype=torch.uint8)
            >>> pixel_values, orig_sizes = preprocessor(img_uint8)
            >>> pixel_values.shape
            torch.Size([2, 3, 1008, 1008])
            >>> orig_sizes.shape
            torch.Size([2, 2])
            >>> # float input
            >>> img_float = torch.rand(2, 3, 480, 640, dtype=torch.float32)
            >>> pixel_values, orig_sizes = preprocessor(img_float)
        """
        assert pixel_values.ndim == 4, f"Expected BCHW tensor, got shape {pixel_values.shape}"

        # Convert uint8 to float if needed
        if pixel_values.dtype == torch.uint8:
            pixel_values = pixel_values.float() / 255.0

        # Get original sizes
        batch_size = pixel_values.shape[0]
        original_height = pixel_values.shape[2]
        original_width = pixel_values.shape[3]
        original_sizes = torch.tensor(
            [[original_height, original_width]] * batch_size,
            dtype=torch.int32,
            device=pixel_values.device,
        )

        # Resize to target size (maintain aspect ratio)
        new_h, new_w = self.get_preprocess_shape(original_height, original_width, self.target_size)
        pixel_values = functional.interpolate(
            pixel_values,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )

        # Pad to target_size x target_size (pad right and bottom)
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        pixel_values = functional.pad(pixel_values, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        # Normalize with ImageNet mean and std
        pixel_values = (pixel_values - self.mean) / self.std

        return pixel_values, original_sizes
