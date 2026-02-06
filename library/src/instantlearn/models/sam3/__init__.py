# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM3 model package.

This package contains the SAM3 (Segment Anything Model 3) implementation
for text and visual prompting segmentation.
"""

from .model import Sam3Model
from .processing import Sam3Postprocessor, Sam3Preprocessor, Sam3PromptPreprocessor
from .sam3 import SAM3

__all__ = [
    "SAM3",
    "Sam3Model",
    "Sam3Postprocessor",
    "Sam3Preprocessor",
    "Sam3PromptPreprocessor",
]
