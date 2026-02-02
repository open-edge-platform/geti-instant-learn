#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from instantlearn.data.base.batch import Batch
from instantlearn.models.matcher import Matcher
from instantlearn.models.per_dino import PerDino
from instantlearn.models.soft_matcher import SoftMatcher

from domain.services.schemas.processor import MatcherConfig, ModelConfig, PerDinoConfig, SoftMatcherConfig
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
                    compile_models=config.compile_models,
                    use_nms=config.use_nms,
                )
                return InferenceModelHandler(model, reference_batch)
            case PerDinoConfig() as config:
                model = PerDino(
                    sam=config.sam_model,
                    encoder_model=config.encoder_model,
                    num_foreground_points=config.num_foreground_points,
                    num_background_points=config.num_background_points,
                    num_grid_cells=config.num_grid_cells,
                    point_selection_threshold=config.point_selection_threshold,
                    confidence_threshold=config.confidence_threshold,
                    use_nms=config.use_nms,
                    precision=config.precision,
                    compile_models=config.compile_models,
                    device=settings.device,
                )
                return InferenceModelHandler(model, reference_batch)
            case SoftMatcherConfig() as config:
                model = SoftMatcher(
                    sam=config.sam_model,
                    encoder_model=config.encoder_model,
                    num_foreground_points=config.num_foreground_points,
                    num_background_points=config.num_background_points,
                    confidence_threshold=config.confidence_threshold,
                    use_sampling=config.use_sampling,
                    use_spatial_sampling=config.use_spatial_sampling,
                    approximate_matching=config.approximate_matching,
                    softmatching_score_threshold=config.softmatching_score_threshold,
                    softmatching_bidirectional=config.softmatching_bidirectional,
                    use_nms=config.use_nms,
                    precision=config.precision,
                    compile_models=config.compile_models,
                    device=settings.device,
                )
                return InferenceModelHandler(model, reference_batch)
            case _:
                return PassThroughModelHandler()
