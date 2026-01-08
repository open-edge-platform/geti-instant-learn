# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientSAM3 - Efficient Student Version of SAM3.

This module provides efficient student versions of SAM3 models using
lightweight backbones (EfficientViT, RepViT, TinyViT) instead of the full ViT.

Adapted from: https://github.com/SimonZeng7108/efficientsam3
"""

from .efficientsam3_image import EfficientSAM3Image
from .model_builder import (
    EfficientSAM3BackboneType,
    EfficientSAM3TextEncoderType,
    ImageStudentEncoder,
    build_efficientsam3_image_model,
)

__all__ = [
    "EfficientSAM3BackboneType",
    "EfficientSAM3Image",
    "EfficientSAM3TextEncoderType",
    "ImageStudentEncoder",
    "build_efficientsam3_image_model",
]
