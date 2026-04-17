# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM3 model package.

This package contains the SAM3 (Segment Anything Model 3) implementation
for text and visual prompting segmentation.
"""

from .model import Sam3Model
from .post_processing import PostProcessingConfig
from .processing import Sam3Postprocessor, Sam3Preprocessor, Sam3PromptPreprocessor
from .sam3 import SAM3, Sam3PromptMode
from .sam3_openvino import SAM3OpenVINO, SAM3OVVariant

__all__ = [
    "SAM3",
    "PostProcessingConfig",
    "SAM3OVVariant",
    "SAM3OpenVINO",
    "Sam3Model",
    "Sam3Postprocessor",
    "Sam3Preprocessor",
    "Sam3PromptMode",
    "Sam3PromptPreprocessor",
]
