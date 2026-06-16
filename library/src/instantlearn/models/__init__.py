# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Models.

Each model is organised in its own self-contained folder with model-specific
components.  Shared components remain in the top-level ``components/``
directory.

New base hierarchy (RFC-01):

* ``Model`` (ABC, torch-free)

  * ``TorchModel``   (Model, nn.Module)
  * ``OpenVINOModel`` (Model, OV-only)
"""

from .base import Model
from .dinotxt import DinoTxtZeroShotClassification
from .efficient_sam3 import EfficientSAM3
from .grounded_sam import GroundedSAM
from .matcher import Matcher
from .model_card import ModelCard
from .openvino_base import OpenVINOModel
from .per_dino import PerDino
from .sam3 import SAM3, SAM3OpenVINO, SAM3OVVariant, Sam3PromptMode
from .soft_matcher import SoftMatcher
from .torch_base import TorchModel

__all__ = [
    "SAM3",
    "DinoTxtZeroShotClassification",
    "EfficientSAM3",
    "GroundedSAM",
    "Matcher",
    "Model",
    "ModelCard",
    "OpenVINOModel",
    "PerDino",
    "SAM3OVVariant",
    "SAM3OpenVINO",
    "Sam3PromptMode",
    "SoftMatcher",
    "TorchModel",
]
