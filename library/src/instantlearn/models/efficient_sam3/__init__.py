# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientSAM3: lightweight student distilled variant of SAM3.

Replaces SAM3's heavy ViT backbone and CLIP text encoder with efficient
alternatives (timm backbones + MobileCLIP-S1) while maintaining the same
detection and segmentation pipeline.
- EfficientSAM3Model: Inherits from SAM3Model and replaces the backbone and text encoder.
- EfficientSAM3: Uses EfficientSAM3Model and provides high level interface for predict (and fit).
"""

from instantlearn.models.sam3.post_processing import PostProcessingConfig
from instantlearn.models.sam3.sam3 import Sam3PromptMode

from .efficient_sam3 import EfficientSAM3
from .model import EfficientSam3Model

__all__ = ["EfficientSAM3", "EfficientSam3Model", "PostProcessingConfig", "Sam3PromptMode"]
