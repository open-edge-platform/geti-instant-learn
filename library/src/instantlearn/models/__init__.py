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

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .base import Model
from .model_card import ModelCard

if TYPE_CHECKING:
    from .dinotxt import DinoTxtZeroShotClassification
    from .efficient_sam3 import EfficientSAM3
    from .grounded_sam import GroundedSAM
    from .matcher import Matcher, MatcherOpenVINO
    from .model_loader import ModelLoader, resolve_model_dir
    from .openvino_base import OpenVINOModel
    from .per_dino import PerDino
    from .sam3 import SAM3, SAM3OpenVINO, SAM3OVVariant, Sam3PromptMode
    from .soft_matcher import SoftMatcher
    from .torch_base import ExportConfig, TorchModel

_EXPORTS = {
    "DinoTxtZeroShotClassification": (".dinotxt", "DinoTxtZeroShotClassification"),
    "EfficientSAM3": (".efficient_sam3", "EfficientSAM3"),
    "ExportConfig": (".torch_base", "ExportConfig"),
    "GroundedSAM": (".grounded_sam", "GroundedSAM"),
    "Matcher": (".matcher", "Matcher"),
    "MatcherOpenVINO": (".matcher", "MatcherOpenVINO"),
    "ModelLoader": (".model_loader", "ModelLoader"),
    "OpenVINOModel": (".openvino_base", "OpenVINOModel"),
    "PerDino": (".per_dino", "PerDino"),
    "SAM3": (".sam3", "SAM3"),
    "SAM3OVVariant": (".sam3", "SAM3OVVariant"),
    "SAM3OpenVINO": (".sam3", "SAM3OpenVINO"),
    "Sam3PromptMode": (".sam3", "Sam3PromptMode"),
    "SoftMatcher": (".soft_matcher", "SoftMatcher"),
    "TorchModel": (".torch_base", "TorchModel"),
    "resolve_model_dir": (".model_loader", "resolve_model_dir"),
}


def __getattr__(name: str) -> object:
    """Resolve concrete model exports only when a caller requests them.

    Raises:
        AttributeError: If *name* is not a public model export.
    """
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg) from error

    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "SAM3",
    "DinoTxtZeroShotClassification",
    "EfficientSAM3",
    "ExportConfig",
    "GroundedSAM",
    "Matcher",
    "MatcherOpenVINO",
    "Model",
    "ModelCard",
    "ModelLoader",
    "OpenVINOModel",
    "PerDino",
    "SAM3OVVariant",
    "SAM3OpenVINO",
    "Sam3PromptMode",
    "SoftMatcher",
    "TorchModel",
    "resolve_model_dir",
]
