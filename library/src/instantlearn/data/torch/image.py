# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch-backed image and mask loaders.

These return torch tensors and live in the torch subpackage so the
backend-neutral contract in :mod:`instantlearn.data` never imports torch.
The numpy readers in :mod:`instantlearn.data.utils.image` do the actual file
I/O; this module only wraps their output as tensors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torchvision import tv_tensors

from instantlearn.data.utils.image import read_image as _read_image_np
from instantlearn.data.utils.image import read_mask as _read_mask_np

if TYPE_CHECKING:
    from pathlib import Path


def read_image(path: str | Path) -> tv_tensors.Image:
    """Read an image as a ``torchvision`` ``tv_tensors.Image`` in CHW format.

    Args:
        path: Local file path or URL to the image.

    Returns:
        The image as a ``tv_tensors.Image`` tensor of shape ``(C, H, W)``.
    """
    array = _read_image_np(path)  # HWC uint8
    chw = np.ascontiguousarray(array.transpose(2, 0, 1))
    return tv_tensors.Image(torch.from_numpy(chw))


def read_mask(path: str | Path) -> torch.Tensor:
    """Read a binarized mask as a ``torch.Tensor`` in HW format.

    Args:
        path: Local file path or URL to the mask.

    Returns:
        The binarized mask as a ``torch.Tensor`` of shape ``(H, W)`` with
        values in ``{0, 1}``.
    """
    array = _read_mask_np(path)  # HW uint8
    return torch.from_numpy(np.ascontiguousarray(array))
