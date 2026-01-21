# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientSAM3 - Efficient Student Version of SAM3.

This module provides efficient student versions of SAM3 models using
lightweight backbones (EfficientViT, RepViT, TinyViT) instead of the full ViT.

All 9 official EfficientSAM3 models are supported:
    - EfficientViT-B0, B1, B2 (0.68M, 4.64M, 14.98M params)
    - RepViT-M0.9, M1.1, M2.3 (4.72M, 7.77M, 22.40M params)
    - TinyViT-5M, 11M, 21M (5.07M, 10.55M, 20.62M params)

Text encoders:
    - MobileCLIP-S0, S1, B, MobileCLIP2-L
    - Full SAM3 text encoder (when text_encoder_type=None)

References:
    Paper: https://arxiv.org/abs/2501.06950
    Repo: https://github.com/SimonZeng7108/efficientsam3
    Models: https://huggingface.co/Simon7108528/EfficientSAM3
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
