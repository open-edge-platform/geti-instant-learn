# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Models."""

from .dinotxt import DinoTextEncoder
from .grounding_dino import GroundingDinoForObjectDetection

__all__ = [
    "DinoTextEncoder",
    "GroundingDinoForObjectDetection",
]
