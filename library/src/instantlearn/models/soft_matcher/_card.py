# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Static capability descriptor for the SoftMatcher model family."""

from instantlearn.device import DeviceType
from instantlearn.models.model_card import ModelCard, RuntimeCapability
from instantlearn.utils.constants import Backend, PromptType, ShotMode

_SOFT_MATCHER_CARD = ModelCard(
    name="SoftMatcher",
    family="soft_matcher",
    description="One-shot segmentation via probabilistic soft feature matching (DINOv3 + SAM decoder)",
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
