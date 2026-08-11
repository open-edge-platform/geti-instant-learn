# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Static capability descriptor for the SAM3 model family."""

from instantlearn.device import DeviceType
from instantlearn.models.model_card import ModelCard, RuntimeCapability
from instantlearn.utils.constants import Backend, PromptType, ShotMode

_SAM3_CARD = ModelCard(
    name="SAM3",
    family="sam3",
    description="Segment Anything 3 model for text, box, point, and visual-exemplar prompting.",
    prompt_types=frozenset({PromptType.TEXT, PromptType.MASK, PromptType.BOUNDING_BOX, PromptType.POINT}),
    shot_modes=frozenset({ShotMode.ZERO_SHOT, ShotMode.ONE_SHOT, ShotMode.FEW_SHOT}),
    exportable_to=frozenset({Backend.OPENVINO, Backend.ONNX}),
    supported_runtimes=frozenset(
        {
            RuntimeCapability(Backend.TORCH, frozenset({DeviceType.CPU, DeviceType.GPU})),
            RuntimeCapability(
                Backend.OPENVINO,
                frozenset({DeviceType.CPU, DeviceType.GPU, DeviceType.NPU}),
            ),
        },
    ),
)
