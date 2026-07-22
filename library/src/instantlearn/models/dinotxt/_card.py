# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Static capability descriptor for the DinoTxt model family."""

from instantlearn.models.model_card import ModelCard
from instantlearn.utils.constants import PromptType, ShotMode

_DINOTXT_CARD = ModelCard(
    name="DinoTxt",
    family="dinotxt",
    description="Zero-shot classification via DINOv3 + text templates",
    prompt_types=frozenset({PromptType.TEXT}),
    shot_modes=frozenset({ShotMode.ZERO_SHOT}),
    exportable_to=frozenset(),
)
