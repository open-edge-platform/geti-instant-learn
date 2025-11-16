#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from getiprompt.models.base import Model
from getiprompt.models.matcher import Matcher

from runtime.core.components.schemas.processor import MatcherConfig, ModelConfig


class ModelFactory:
    """
    A factory for creating Model instances based on a configuration.
    """

    @classmethod
    def create(cls, config: ModelConfig | None) -> Model | None:
        match config:
            case MatcherConfig() as config:
                return Matcher(
                    num_foreground_points=config.num_foreground_points,
                    num_background_points=config.num_background_points,
                    mask_similarity_threshold=config.mask_similarity_threshold,
                    precision=config.precision,
                    device="cpu",
                )
            case _:
                return None
