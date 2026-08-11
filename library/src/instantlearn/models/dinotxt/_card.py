# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Static capability descriptor for the DinoTxt model family."""

from instantlearn.device import DeviceType
from instantlearn.models.model_card import ModelCard, RuntimeCapability
from instantlearn.utils.constants import Backend, PromptType, ShotMode

_DINOTXT_CARD = ModelCard(
    name="DinoTxt",
    family="dinotxt",
    description="Zero-shot classification via DINOv3 + text templates",
    prompt_types=frozenset({PromptType.TEXT}),
    shot_modes=frozenset({ShotMode.ZERO_SHOT}),
    exportable_to=frozenset(),
    supported_runtimes=frozenset({RuntimeCapability(Backend.TORCH, frozenset({DeviceType.CPU, DeviceType.GPU}))}),
)
