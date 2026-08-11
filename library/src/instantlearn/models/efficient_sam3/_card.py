# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Static capability descriptor for the EfficientSAM3 model."""

from instantlearn.models.model_card import ModelCard
from instantlearn.utils.constants import Backend, PromptType, ShotMode

# EfficientSAM3 subclasses SAM3 but is a distinct model: distilled lightweight
# backbones with a MobileCLIP text encoder. It therefore carries its own card
# rather than inheriting SAM3's, which would report the wrong name and family.
_EFFICIENT_SAM3_CARD = ModelCard(
    name="EfficientSAM3",
    family="efficient_sam3",
    description="Distilled SAM3 with lightweight student backbones for text and visual prompting.",
    prompt_types=frozenset({PromptType.TEXT, PromptType.MASK, PromptType.BOUNDING_BOX, PromptType.POINT}),
    shot_modes=frozenset({ShotMode.ZERO_SHOT, ShotMode.ONE_SHOT, ShotMode.FEW_SHOT}),
    exportable_to=frozenset({Backend.OPENVINO, Backend.ONNX}),
)
