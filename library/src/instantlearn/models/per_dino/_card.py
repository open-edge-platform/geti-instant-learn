# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Static capability descriptor for the PerDINO model family."""

from instantlearn.models.model_card import ModelCard
from instantlearn.utils.constants import Backend, PromptType, ShotMode

_PERDINO_CARD = ModelCard(
    name="PerDINO",
    family="perdino",
    description="One-shot segmentation via DINOv3 cosine-similarity grid prompting + SAM decoder",
    prompt_types=frozenset({PromptType.MASK}),
    shot_modes=frozenset({ShotMode.ONE_SHOT, ShotMode.FEW_SHOT}),
    exportable_to=frozenset({Backend.OPENVINO, Backend.ONNX}),
)
