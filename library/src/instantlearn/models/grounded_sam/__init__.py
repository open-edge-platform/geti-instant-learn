# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""GroundedSAM model package.

This package contains the GroundedSAM model implementation that uses
a zero-shot object detector to generate boxes for SAM segmentation.
"""

from .grounded import GroundingModel, TextToBoxPromptGenerator
from .grounded_sam import GroundedSAM
from .grounding_dino import GroundingDinoForObjectDetection
from .prompt_filter import BoxPromptFilter

__all__ = [
    "BoxPromptFilter",
    "GroundedSAM",
    "GroundingDinoForObjectDetection",
    "GroundingModel",
    "TextToBoxPromptGenerator",
]
