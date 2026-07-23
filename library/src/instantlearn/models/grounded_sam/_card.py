# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Static capability descriptor for the GroundedSAM model family."""

from instantlearn.models.model_card import ModelCard
from instantlearn.utils.constants import PromptType, ShotMode

_GROUNDED_SAM_CARD = ModelCard(
    name="GroundedSAM",
    family="grounded_sam",
    description="Zero-shot object detection via text grounding + SAM",
    prompt_types=frozenset({PromptType.TEXT}),
    shot_modes=frozenset({ShotMode.ZERO_SHOT}),
    exportable_to=frozenset(),
)
