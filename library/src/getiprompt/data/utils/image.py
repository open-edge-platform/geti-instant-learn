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
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import Image, Mask


def read_image(path: str | Path, as_tensor: bool = True) -> Image | np.ndarray:
    """Read an image from file.

    Args:
        path (str | Path): Path to the image file.
        as_tensor (bool, optional): Whether to return as tensor. Defaults to ``True``.

    Returns:
        Image | np.ndarray: Loaded image as tensor (CHW format) or numpy array (HWC format).

    Note:
            - When as_tensor=True: Returns CHW format (C, H, W)
            - When as_tensor=False: Returns HWC format (H, W, C)

            This is intentional - models expect HWC format for preprocessing.
            The model preprocessors (HuggingFace, SAM) handle the channel permutation internally.

    Raises:
        FileNotFoundError: If the image file does not exist.

    Example:
        >>> image = read_image("path/to/image.jpg")
        >>> image.shape
        torch.Size([3, 224, 224])

        >>> # As numpy array (HWC format for model preprocessors)
        >>> image_np = read_image("path/to/image.jpg", as_tensor=False)
        >>> image_np.shape
        (224, 224, 3)
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
    # Return as numpy array in HWC format (uint8, 0-255 range)
    # Models expect HWC format for preprocessing (HuggingFace, SAM transforms)
    return np.array(pil_image, dtype=np.uint8)


def read_mask(path: str | Path, as_tensor: bool = True) -> Mask | np.ndarray:
    """Read a mask from file.

    Args:
        path (str | Path): Path to the mask file.
        as_tensor (bool, optional): Whether to return as tensor. Defaults to ``True``.

    Returns:
        Mask | np.ndarray: Loaded mask as tensor (HW format) or numpy array (HW format).

    Note:
            The mask is binarized to 0 (background) and 1 (foreground).
            Any non-zero pixel value in the input is treated as foreground.

    Raises:
        FileNotFoundError: If the mask file does not exist.

    Example:
        >>> mask = read_mask("path/to/mask.png")
        >>> mask.shape
        torch.Size([224, 224])
        >>> np.unique(mask.numpy())
        array([0, 1])

        >>> # As numpy array
        >>> mask_np = read_mask("path/to/mask.png", as_tensor=False)
        >>> mask_np.shape
        (224, 224)
        >>> np.unique(mask_np)
        array([0, 1])
    """
    path = Path(path)
    if not path.exists():
        msg = f"Mask file not found: {path}"
        raise FileNotFoundError(msg)

    pil_image = PILImage.open(path).convert("L")  # Convert to grayscale
    mask_array = np.array(pil_image, dtype=np.uint8)

    # Binarize: any non-zero value becomes 1
    binary_array = (mask_array > 0).astype(np.uint8)

    if as_tensor:
        # Convert to tensor
        binary_tensor = torch.from_numpy(binary_array)
        return Mask(binary_tensor)
    return binary_array
