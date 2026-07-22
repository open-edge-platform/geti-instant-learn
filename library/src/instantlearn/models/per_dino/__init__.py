# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PerDino model package.

This package contains the PerDino algorithm model for one-shot segmentation,
which matches reference objects using DINOv2 features and cosine similarity.
"""

from .per_dino import PerDino, PerDinoInferenceGraph
from .per_dino_openvino import PerDinoOpenVINO
from .prompt_generators import GridPromptGenerator

__all__ = [
    "GridPromptGenerator",
    "PerDino",
    "PerDinoInferenceGraph",
    "PerDinoOpenVINO",
]
