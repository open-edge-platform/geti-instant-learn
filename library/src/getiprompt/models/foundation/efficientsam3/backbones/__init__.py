# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientSAM3 backbone implementations.

This module provides student backbones for EfficientSAM3:
    - EfficientViT: Efficient vision transformers with lite multi-head attention
    - RepViT: Reparameterizable vision transformers
    - TinyViT: Compact vision transformers
    - MobileCLIP: Efficient text encoders
"""

from .efficientvit import EfficientViT, efficientvit_b0, efficientvit_b1, efficientvit_b2
from .mobile_clip import MobileCLIPTextTransformer, TextStudentEncoder
from .repvit import RepViT, repvit_m0_9, repvit_m1_1, repvit_m2_3
from .tiny_vit import TinyViT, tiny_vit_5m_224, tiny_vit_11m_224, tiny_vit_21m_224

__all__ = [
    # EfficientViT
    "EfficientViT",
    "efficientvit_b0",
    "efficientvit_b1",
    "efficientvit_b2",
    # RepViT
    "RepViT",
    "repvit_m0_9",
    "repvit_m1_1",
    "repvit_m2_3",
    # TinyViT
    "TinyViT",
    "tiny_vit_5m_224",
    "tiny_vit_11m_224",
    "tiny_vit_21m_224",
    # MobileCLIP Text Encoder
    "MobileCLIPTextTransformer",
    "TextStudentEncoder",
]
