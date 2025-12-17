#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import os

from getiprompt.data.base.batch import Batch
from getiprompt.models.matcher import Matcher

from domain.services.schemas.processor import MatcherConfig, ModelConfig
from runtime.core.components.base import ModelHandler
from runtime.core.components.models.inference_model import InferenceModelHandler
from runtime.core.components.models.passthrough_model import PassThroughModelHandler

DEVICE_MAP = {
    "cpu": "cpu",  # OpenVINO on CPU
    "gpu": "cuda",  # Torch with NVIDIA GPU
    "xpu": "xpu",  # Torch with Intel GPU
}


class ModelFactory:
    @staticmethod
    def _resolve_device() -> str:
        """Resolve the device based on RUNTIME environment variable."""
        runtime = os.getenv("RUNTIME", "cpu").lower()
        device = DEVICE_MAP.get(runtime)
        if device is None:
            raise ValueError(f"Unknown runtime: {runtime}")
        return device

    @classmethod
    def create(cls, reference_batch: Batch | None, config: ModelConfig | None) -> ModelHandler:
        if reference_batch is None:
            return PassThroughModelHandler()
        match config:
            case MatcherConfig() as config:
                model = Matcher(
                    num_foreground_points=config.num_foreground_points,
                    num_background_points=config.num_background_points,
                    mask_similarity_threshold=config.mask_similarity_threshold,
                    precision=config.precision,
                    device=cls._resolve_device(),
                )
                return InferenceModelHandler(model, reference_batch)
            case _:
                return PassThroughModelHandler()
