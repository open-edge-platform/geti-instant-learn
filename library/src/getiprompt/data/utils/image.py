# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Image utilities for GetiPrompt datasets.

This module provides functions for reading and processing images and masks
for GetiPrompt few-shot segmentation tasks.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.tv_tensors import Image, Mask
from torchvision.transforms.v2 import functional as F


def read_image(path: str | Path, as_tensor: bool = True) -> Image | np.ndarray:
    """Read an image from file.

    Args:
        path (str | Path): Path to the image file.
        as_tensor (bool, optional): Whether to return as tensor. Defaults to ``True``.

    Returns:
        Image | np.ndarray: Loaded image as tensor (CHW format) or numpy array (HWC format).

    Raises:
        FileNotFoundError: If the image file does not exist.

    Example:
        >>> image = read_image("path/to/image.jpg")
        >>> image.shape
        torch.Size([3, 224, 224])
        
        >>> # As numpy array
        >>> image_np = read_image("path/to/image.jpg", as_tensor=False)
        >>> image_np.shape
        (3, 224, 224)
    """
    path = Path(path)
    if not path.exists():
        msg = f"Image file not found: {path}"
        raise FileNotFoundError(msg)

    pil_image = PILImage.open(path).convert("RGB")
    
    if as_tensor:
        # Convert to tensor and ensure it's in CHW format
        tensor = F.to_tensor(pil_image)
        return Image(tensor)
    else:
        # Return as numpy array in CHW format (uint8, 0-255 range) for consistency with tensors
        array_hwc = np.array(pil_image, dtype=np.uint8)
        array_chw = np.transpose(array_hwc, (2, 0, 1))  # Convert HWC to CHW
        return array_chw


def read_mask(path: str | Path, as_tensor: bool = True) -> Mask | np.ndarray:
    """Read a mask from file.

    Args:
        path (str | Path): Path to the mask file.
        as_tensor (bool, optional): Whether to return as tensor. Defaults to ``True``.

    Returns:
        Mask | np.ndarray: Loaded mask as tensor (HW format) or numpy array (HW format).

    Raises:
        FileNotFoundError: If the mask file does not exist.

    Example:
        >>> mask = read_mask("path/to/mask.png")
        >>> mask.shape
        torch.Size([224, 224])
        
        >>> # As numpy array
        >>> mask_np = read_mask("path/to/mask.png", as_tensor=False)
        >>> mask_np.shape
        (224, 224)
    """
    path = Path(path)
    if not path.exists():
        msg = f"Mask file not found: {path}"
        raise FileNotFoundError(msg)

    pil_image = PILImage.open(path).convert("L")  # Convert to grayscale
    
    if as_tensor:
        # Convert to tensor and ensure it's binary (0 or 1)
        tensor = F.to_tensor(pil_image).squeeze(0)  # Remove channel dimension
        binary_tensor = (tensor > 0.5).to(torch.uint8)  # Threshold to binary
        return Mask(binary_tensor)
    else:
        # Return as numpy array (uint8, 0-255 range, then threshold to binary)
        mask_array = np.array(pil_image, dtype=np.uint8)
        binary_array = (mask_array > 127).astype(np.uint8)  # Threshold to binary
        return binary_array
