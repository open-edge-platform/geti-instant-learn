# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Models.

Each model is organized in its own self-contained folder with model-specific components.
Shared components remain in the top-level components/ directory.
"""

from .base import Model
from .dinotxt import DinoTxtZeroShotClassification
from .efficient_sam3 import EfficientSAM3
from .grounded_sam import GroundedSAM
from .matcher import Matcher
from .per_dino import PerDino
from .sam3 import SAM3, Sam3PromptMode
from .soft_matcher import SoftMatcher

__all__ = [
    "SAM3",
    "DinoTxtZeroShotClassification",
    "EfficientSAM3",
    "GroundedSAM",
    "Matcher",
    "Model",
    "PerDino",
    "Sam3PromptMode",
    "SoftMatcher",
]
