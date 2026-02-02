# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""SAM3 model components for HuggingFace Transformers.

This module provides the main classes for SAM3 (Segment Anything Model 3):
- `Sam3Model`: The main SAM3 model for open-vocabulary instance segmentation
- `ImageProcessorFast`: Fast image processing for SAM3 inputs
- `Processor`: Unified processor for images and text inputs
"""

from .image_processing import ImageProcessorFast
from .model import Sam3Model
from .processing import Processor

__all__ = [
    "ImageProcessorFast",
    "Processor",
    "Sam3Model",
]
