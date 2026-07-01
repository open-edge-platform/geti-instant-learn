# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Models.

Each model is organized in its own self-contained folder with model-specific
components. Shared components live in the top-level ``components/`` directory.

The model hierarchy is:

- ``Model`` (ABC, torch-free)
    - ``TorchModel`` (nn.Module, Model)
    - ``OpenVINOModel`` (Model, OV-only)
"""

from .base import Model
from .dinotxt import DinoTxtZeroShotClassification
from .efficient_sam3 import EfficientSAM3
from .grounded_sam import GroundedSAM
from .matcher import Matcher
from .model_card import ModelCard
from .model_loader import ModelLoader, resolve_model_dir
from .openvino_base import OpenVINOModel
from .per_dino import PerDino
from .sam3 import SAM3, SAM3OpenVINO, Sam3PromptMode
from .soft_matcher import SoftMatcher
from .torch_base import ExportConfig, TorchModel

__all__ = [
    "SAM3",
    "DinoTxtZeroShotClassification",
    "EfficientSAM3",
    "ExportConfig",
    "GroundedSAM",
    "Matcher",
    "Model",
    "ModelCard",
    "ModelLoader",
    "OpenVINOModel",
    "PerDino",
    "SAM3OpenVINO",
    "Sam3PromptMode",
    "SoftMatcher",
    "TorchModel",
    "resolve_model_dir",
]
