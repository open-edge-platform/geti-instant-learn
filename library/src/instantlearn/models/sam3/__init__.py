# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM3 model package.

This package contains the SAM3 (Segment Anything Model 3) implementation
for text and visual prompting segmentation.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Sam3Model
    from .post_processing import PostProcessingConfig
    from .processing import Sam3Postprocessor, Sam3Preprocessor, Sam3PromptPreprocessor
    from .sam3 import SAM3, SAM3_APPLICATION_MODEL_ID, SAM3_LIBRARY_MODEL_ID, CanvasConfig, Sam3PromptMode
    from .sam3_openvino import SAM3OpenVINO, SAM3OVVariant

_EXPORTS = {
    "CanvasConfig": (".sam3", "CanvasConfig"),
    "PostProcessingConfig": (".post_processing", "PostProcessingConfig"),
    "SAM3": (".sam3", "SAM3"),
    "SAM3_APPLICATION_MODEL_ID": (".sam3", "SAM3_APPLICATION_MODEL_ID"),
    "SAM3_LIBRARY_MODEL_ID": (".sam3", "SAM3_LIBRARY_MODEL_ID"),
    "SAM3OVVariant": (".sam3_openvino", "SAM3OVVariant"),
    "SAM3OpenVINO": (".sam3_openvino", "SAM3OpenVINO"),
    "Sam3Model": (".model", "Sam3Model"),
    "Sam3Postprocessor": (".processing", "Sam3Postprocessor"),
    "Sam3Preprocessor": (".processing", "Sam3Preprocessor"),
    "Sam3PromptMode": (".sam3", "Sam3PromptMode"),
    "Sam3PromptPreprocessor": (".processing", "Sam3PromptPreprocessor"),
}


def __getattr__(name: str) -> object:
    """Resolve SAM3 implementation exports only when requested.

    Raises:
        AttributeError: If *name* is not a public SAM3 export.
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
    "SAM3_APPLICATION_MODEL_ID",
    "SAM3_LIBRARY_MODEL_ID",
    "CanvasConfig",
    "PostProcessingConfig",
    "SAM3OVVariant",
    "SAM3OpenVINO",
    "Sam3Model",
    "Sam3Postprocessor",
    "Sam3Preprocessor",
    "Sam3PromptMode",
    "Sam3PromptPreprocessor",
]
