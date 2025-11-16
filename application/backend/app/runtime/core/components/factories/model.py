#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import os

from getiprompt.models.base import Model
from getiprompt.models.matcher import Matcher

from runtime.core.components.schemas.processor import MatcherConfig, ModelConfig

DEVICE_MAP = {
    "cpu": "cpu",  # OpenVINO on CPU
    "cuda": "cuda",  # Torch with NVIDIA GPU
    "xpu": "xpu",  # Torch with Intel GPU
}


class ModelFactory:
    """
    A factory for creating Model instances based on a configuration.
    """

    @staticmethod
    def _resolve_device() -> str:
        """Resolve the device based on RUNTIME environment variable.

        Returns:
            The device string (cpu, cuda, or xpu).

        Raises:
            ValueError: If the runtime is not supported.
        """
        runtime = os.getenv("RUNTIME", "cpu").lower()
        device = DEVICE_MAP.get(runtime)
        if device is None:
            raise ValueError(f"Unknown runtime: {runtime}")
        return device

    @classmethod
    def create(cls, config: ModelConfig | None) -> Model | None:
        match config:
            case MatcherConfig() as config:
                return Matcher(
                    num_foreground_points=config.num_foreground_points,
                    num_background_points=config.num_background_points,
                    mask_similarity_threshold=config.mask_similarity_threshold,
                    precision=config.precision,
                    device=cls._resolve_device(),
                )
            case _:
                return None
