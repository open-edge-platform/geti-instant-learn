# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utils."""

from .constants import HUGGINGFACE_AVAILABLE_IMAGE_ENCODERS, TIMM_AVAILABLE_IMAGE_ENCODERS, Backend
from .similarity_resize import resize_similarity_maps
from .utils import (
    download_file,
    precision_to_torch_dtype,
    setup_logger,
)

_IMPORTING_FROM_UTILS = True

__all__ = [
    "HUGGINGFACE_AVAILABLE_IMAGE_ENCODERS",
    "TIMM_AVAILABLE_IMAGE_ENCODERS",
    "Backend",
    "download_file",
    "precision_to_torch_dtype",
    "resize_similarity_maps",
    "setup_logger",
]
