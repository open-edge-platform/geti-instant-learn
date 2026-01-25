#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from getiprompt.data.base.batch import Batch
from getiprompt.models.matcher import Matcher

from domain.services.schemas.processor import MatcherConfig, ModelConfig
from runtime.core.components.base import ModelHandler
from runtime.core.components.models.inference_model import InferenceModelHandler
from runtime.core.components.models.passthrough_model import PassThroughModelHandler
from settings import get_settings


class ModelFactory:
    @classmethod
    def create(cls, reference_batch: Batch | None, config: ModelConfig | None) -> ModelHandler:
        if reference_batch is None:
            return PassThroughModelHandler()
        settings = get_settings()
        if not settings.processor_inference_enabled:
            return PassThroughModelHandler()
        match config:
            case MatcherConfig() as config:
                model = Matcher(
                    num_foreground_points=config.num_foreground_points,
                    num_background_points=config.num_background_points,
                    confidence_threshold=config.confidence_threshold,
                    precision=config.precision,
                    device=settings.device,
                    use_mask_refinement=config.use_mask_refinement,
                    sam=config.sam_model,
                    encoder_model=config.encoder_model,
                )
                return InferenceModelHandler(model, reference_batch)
            case _:
                return PassThroughModelHandler()
