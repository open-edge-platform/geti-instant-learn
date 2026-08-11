# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Static capability descriptor for the PerDino model family."""

from instantlearn.device import DeviceType
from instantlearn.models.model_card import ModelCard, RuntimeCapability
from instantlearn.utils.constants import Backend, PromptType, ShotMode

_PERDINO_CARD = ModelCard(
    name="PerDino",
    family="per_dino",
    description="One-shot segmentation via DINOv3 cosine-similarity grid prompting + SAM decoder",
    prompt_types=frozenset({PromptType.MASK}),
    shot_modes=frozenset({ShotMode.ONE_SHOT, ShotMode.FEW_SHOT}),
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
