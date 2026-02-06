# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Matcher model package.

This package contains the Matcher model implementation based on the paper
'Segment Anything with One Shot Using All-Purpose Feature Matching'.
"""

from .prompt_generators import BidirectionalPromptGenerator
from .matcher import (
    EncoderForwardFeaturesWrapper,
    Matcher,
    MatcherInferenceGraph,
)

__all__ = [
    "BidirectionalPromptGenerator",
    "EncoderForwardFeaturesWrapper",
    "Matcher",
    "MatcherInferenceGraph",
]
