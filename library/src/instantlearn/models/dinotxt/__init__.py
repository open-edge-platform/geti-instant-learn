# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOTxt model package.

This package contains the DINOv3 zero-shot classification model
using text encoders for zero-shot image classification.
"""

from .dinotxt import DinoTxtZeroShotClassification
from .encoder import IMAGENET_TEMPLATES, DinoTextEncoder

__all__ = [
    "IMAGENET_TEMPLATES",
    "DinoTextEncoder",
    "DinoTxtZeroShotClassification",
]
