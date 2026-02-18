#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from instantlearn.data.base.batch import Batch
from instantlearn.models.matcher import Matcher
from instantlearn.models.per_dino import PerDino
from instantlearn.models.soft_matcher import SoftMatcher

from domain.services.schemas.processor import MatcherConfig, ModelConfig, PerDinoConfig, SoftMatcherConfig
from runtime.core.components.base import ModelHandler
from runtime.core.components.models.openvino_model import OpenVINOModelHandler
from runtime.core.components.models.passthrough_model import PassThroughModelHandler
from runtime.core.components.models.torch_model import TorchModelHandler
from settings import get_settings


class ModelFactory:
    @classmethod
    def create(cls, reference_batch: Batch | None, config: ModelConfig | None) -> ModelHandler:
        if reference_batch is None:
            return PassThroughModelHandler()
        settings = get_settings()

        is_cuda = settings.device == "cuda"

        if not settings.processor_inference_enabled:
            return PassThroughModelHandler()
        match config:
            case MatcherConfig() as config:
                # if the model is converted to the OV format, the precision should be strictly fp32
                # as a suggestion we can handle conversion at the higher level factory,
                # as it knows if the model should be converted or not and can override the configuration
                # of the model
                precision = config.precision if is_cuda else "fp32"
                model = Matcher(
                    sam=config.sam_model,
                    encoder_model=config.encoder_model,
                    num_foreground_points=config.num_foreground_points,
                    num_background_points=config.num_background_points,
                    confidence_threshold=config.confidence_threshold,
                    use_mask_refinement=config.use_mask_refinement,
                    use_nms=config.use_nms,
                    precision=precision,
                    compile_models=config.compile_models,
                    device=settings.device,
                )
                if is_cuda:
                    return TorchModelHandler(model, reference_batch)
                return OpenVINOModelHandler(model, reference_batch, precision=precision)
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
                return TorchModelHandler(model, reference_batch)
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
                return TorchModelHandler(model, reference_batch)
            case _:
                return PassThroughModelHandler()
