# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SoftMatcher model package.

This package contains the SoftMatcher model implementation based on the paper
'Probabilistic Feature Matching for Fast Scalable Visual Prompting'.
"""

from .prompt_generator import SoftmatcherPromptGenerator
from .soft_matcher import SoftMatcher

__all__ = [
    "SoftMatcher",
    "SoftmatcherPromptGenerator",
]
