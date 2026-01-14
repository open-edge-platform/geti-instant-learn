# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Encoders."""

from .base import ImageEncoder, load_image_encoder
from .huggingface import AVAILABLE_IMAGE_ENCODERS, HuggingFaceImageEncoder
from .timm import AVAILABLE_IMAGE_ENCODERS as TIMM_AVAILABLE_IMAGE_ENCODERS
from .timm import TimmImageEncoder

__all__ = [
    "AVAILABLE_IMAGE_ENCODERS",
    "TIMM_AVAILABLE_IMAGE_ENCODERS",
    "HuggingFaceImageEncoder",
    "ImageEncoder",
    "TimmImageEncoder",
    "load_image_encoder",
]
