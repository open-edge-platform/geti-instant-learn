# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Image utilities for InstantLearn datasets.

This module provides functions for reading and processing images and masks
for InstantLearn few-shot segmentation tasks.
"""

import io
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import tv_tensors


def _is_url(path: str | Path) -> bool:
    """Check if the given path is a URL."""
    try:
        result = urlparse(str(path))
        return result.scheme in {"http", "https"}
    except ValueError:
        return False


def _open_image(path: str | Path, mode: str = "RGB") -> PILImage.Image:
    """Open an image from a local path or URL.

    Args:
        path: Local file path or URL.
        mode: PIL image mode to convert to.

    Returns:
        Opened PIL image converted to the specified mode.

    Raises:
        FileNotFoundError: If a local file does not exist.
    """
    if _is_url(path):
        with urlopen(str(path)) as response:  # noqa: S310
            data = response.read()
        return PILImage.open(io.BytesIO(data)).convert(mode)

    path = Path(path)
    if not path.exists():
        msg = f"Image file not found: {path}"
        raise FileNotFoundError(msg)
    return PILImage.open(path).convert(mode)


def read_image(path: str | Path, as_tensor: bool = True) -> tv_tensors.Image | np.ndarray:
    """Read an image from a local file or URL.

    Args:
        path: Local file path or URL to the image.
        as_tensor: Whether to return as tensor. Defaults to ``True``.

    Returns:
        Loaded image as tensor (CHW format) or numpy array (HWC format).

    Note:
            - When as_tensor=True: Returns CHW format (C, H, W)
            - When as_tensor=False: Returns HWC format (H, W, C)

            This is intentional - models expect HWC format for preprocessing.
            The model preprocessors (HuggingFace, SAM) handle the channel permutation internally.

    Raises:
        FileNotFoundError: If a local image file does not exist.

    Example:
        >>> image = read_image("path/to/image.jpg")
        >>> image.shape
        torch.Size([3, 224, 224])

        >>> # From a URL
        >>> image = read_image("https://example.com/image.jpg")

        >>> # As numpy array (HWC format for model preprocessors)
        >>> image_np = read_image("path/to/image.jpg", as_tensor=False)
        >>> image_np.shape
        (224, 224, 3)
    """
    pil_image = _open_image(path, mode="RGB")

    if as_tensor:
        return tv_tensors.Image(pil_image)
    return np.array(pil_image, dtype=np.uint8)


def read_mask(path: str | Path, as_tensor: bool = True) -> torch.Tensor | np.ndarray:
    """Read a mask from a local file or URL.

    Args:
        path: Local file path or URL to the mask.
        as_tensor: Whether to return as tensor. Defaults to ``True``.

    Returns:
        Loaded mask as tensor (HW format) or numpy array (HW format).

    Note:
            The mask is binarized to 0 (background) and 1 (foreground).
            Any non-zero pixel value in the input is treated as foreground.

    Raises:
        FileNotFoundError: If a local mask file does not exist.

    Example:
        >>> mask = read_mask("path/to/mask.png")
        >>> mask.shape
        torch.Size([224, 224])
        >>> np.unique(mask.numpy())
        array([0, 1])

        >>> # From a URL
        >>> mask = read_mask("https://example.com/mask.png")

        >>> # As numpy array
        >>> mask_np = read_mask("path/to/mask.png", as_tensor=False)
        >>> mask_np.shape
        (224, 224)
        >>> np.unique(mask_np)
        array([0, 1])
    """
    pil_image = _open_image(path, mode="L")
    mask_array = np.array(pil_image, dtype=np.uint8)

    # Binarize: any non-zero value becomes 1
    binary_array = (mask_array > 0).astype(np.uint8)

    if as_tensor:
        # Convert to tensor
        return torch.from_numpy(binary_array)
    return binary_array
