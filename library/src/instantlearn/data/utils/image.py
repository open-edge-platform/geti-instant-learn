# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Numpy image utilities for InstantLearn datasets.

This module is backend-neutral and imports zero torch. It reads images and
masks as numpy arrays. Torch-tensor loaders live in
:mod:`instantlearn.data.torch.image`.
"""

from __future__ import annotations

import io
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
from PIL import Image as PILImage



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


def read_image(path: str | Path) -> np.ndarray:
    """Read an image from a local file or URL as a numpy array.

    Args:
        path: Local file path or URL to the image.

    Returns:
        Loaded image as a numpy array in HWC format (H, W, C), dtype uint8.

    Example:
        >>> image = read_image("path/to/image.jpg")
        >>> image.shape
        (224, 224, 3)

        >>> # From a URL
        >>> image = read_image("https://example.com/image.jpg")
    """
    pil_image = _open_image(path, mode="RGB")
    return np.array(pil_image, dtype=np.uint8)


def read_mask(path: str | Path) -> np.ndarray:
    """Read a mask from a local file or URL as a numpy array.

    Args:
        path: Local file path or URL to the mask.

    Returns:
        Loaded mask as a numpy array in HW format, dtype uint8.

    Note:
            The mask is binarized to 0 (background) and 1 (foreground).
            Any non-zero pixel value in the input is treated as foreground.

    Raises:
        FileNotFoundError: If a local mask file does not exist.

    Example:
        >>> mask = read_mask("path/to/mask.png")
        >>> mask.shape
        (224, 224)
        >>> np.unique(mask)
        array([0, 1])
    """
    pil_image = _open_image(path, mode="L")
    mask_array = np.array(pil_image, dtype=np.uint8)

    # Binarize: any non-zero value becomes 1
    return (mask_array > 0).astype(np.uint8)
