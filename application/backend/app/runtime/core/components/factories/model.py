#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging

from instantlearn.data.base.batch import Batch
from instantlearn.models.matcher import Matcher
from instantlearn.models.model_card import ModelCard
from instantlearn.models.per_dino import PerDino
from instantlearn.models.sam3 import SAM3, Sam3PromptMode
from instantlearn.models.soft_matcher import SoftMatcher
from instantlearn.utils.constants import Backend

from domain.services.schemas.processor import MatcherConfig, ModelConfig, PerDinoConfig, Sam3Config, SoftMatcherConfig
from runtime.core.components.base import ModelHandler
from runtime.core.components.models.openvino_model import OpenVINOModelHandler
from runtime.core.components.models.passthrough_model import PassThroughModelHandler
from runtime.core.components.models.torch_model import TorchModelHandler
from runtime.services.device import DeviceService, ResolvedDevice
from settings import get_settings

logger = logging.getLogger(__name__)


class ModelFactory:
    def __init__(
        self,
        device_service: DeviceService,
    ) -> None:
        self._device_service = device_service

    def _resolve_device(
        self,
        model_card: ModelCard,
        configured_device: str | None,
        allowed_runtimes: tuple[Backend, ...],
    ) -> ResolvedDevice:
        """Resolve a configured preference to a concrete model runtime route."""
        resolved = self._device_service.resolve_for_model(
            model_card=model_card,
            device_str=configured_device or "auto",
            allowed_runtimes=allowed_runtimes,
        )
        logger.info(
            "Accelerator selected: runtime=%s device=%s id=%s (configured=%r, fallback=%s)",
            resolved.runtime.value,
            resolved.device.key,
            resolved.runtime_id,
            configured_device,
            resolved.fallback_used,
        )
        return resolved

    def create(  # noqa: PLR0911
        self,
        reference_batch: Batch | None,
        config: ModelConfig | None,
        configured_device: str | None = None,
    ) -> ModelHandler:
        logger.info("Initializing a model: %s", config)

        if reference_batch is None:
            logger.info("No prompts provided, creating a passthrough model")
            return PassThroughModelHandler()
        settings = get_settings()
        if not settings.processor_inference_enabled:
            logger.info("Inference is disabled in the application settings, creating a passthrough model")
            return PassThroughModelHandler()
        if config is None:
            logger.info("No model config is provided, creating a passthrough model")
            return PassThroughModelHandler()

        match config:
            case MatcherConfig() as config:
                logger.info("Initializing a Matcher instance")
                allowed_runtimes = (
                    (Backend.OPENVINO, Backend.TORCH) if settings.processor_openvino_enabled else (Backend.TORCH,)
                )
                resolved = self._resolve_device(Matcher.card(), configured_device, allowed_runtimes)
                # if the model is converted to the OV format, the precision should be strictly fp32
                # as a suggestion we can handle conversion at the higher level factory,
                # as it knows if the model should be converted or not and can override the configuration
                # of the model
                precision = config.precision if resolved.runtime == Backend.TORCH else "fp32"
                model_device = resolved.device
                if resolved.runtime == Backend.OPENVINO:
                    model_device = self._resolve_device(Matcher.card(), "cpu", (Backend.TORCH,)).device
                model = Matcher(
                    sam=config.sam_model,
                    encoder_model=config.encoder_model,
                    num_foreground_points=config.num_foreground_points,
                    num_background_points=config.num_background_points,
                    confidence_threshold=config.confidence_threshold,
                    use_mask_refinement=config.use_mask_refinement,
                    similarity_threshold=config.similarity_threshold,
                    num_grid_cells=config.num_grid_cells,
                    precision=precision,
                    device=model_device,
                )
                if resolved.runtime == Backend.OPENVINO:
                    logger.info("Using the OpenVINO backend for Matcher")
                    return OpenVINOModelHandler(
                        model=model,
                        reference_batch=reference_batch,
                        precision=precision,
                        ov_device=resolved.runtime_id,
                        compression_preset=config.preset,
                    )
                logger.info("Using the Torch backend for Matcher")
                return TorchModelHandler(model, reference_batch)
            case PerDinoConfig() as config:
                logger.info("Initializing a PerDINO instance")
                resolved = self._resolve_device(PerDino.card(), configured_device, (Backend.TORCH,))
                model = PerDino(
                    sam=config.sam_model,
                    encoder_model=config.encoder_model,
                    num_foreground_points=config.num_foreground_points,
                    num_background_points=config.num_background_points,
                    num_grid_cells=config.num_grid_cells,
                    point_selection_threshold=config.point_selection_threshold,
                    confidence_threshold=config.confidence_threshold,
                    precision=config.precision,
                    device=resolved.device,
                )
                logger.info("Using the Torch backend for PerDINO")
                return TorchModelHandler(model, reference_batch)
            case SoftMatcherConfig() as config:
                logger.info("Initializing a SoftMatcher instance")
                resolved = self._resolve_device(SoftMatcher.card(), configured_device, (Backend.TORCH,))
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
                    precision=config.precision,
                    device=resolved.device,
                )
                logger.info("Using the Torch backend for SoftMatcher")
                return TorchModelHandler(model, reference_batch)
            case Sam3Config() as config:
                logger.info("Initializing a SAM3 instance")
                resolved = self._resolve_device(SAM3.card(), configured_device, (Backend.TORCH,))
                has_bboxes = any(s.bboxes is not None for s in reference_batch.samples)
                prompt_mode = Sam3PromptMode.CANVAS if has_bboxes else Sam3PromptMode.CLASSIC
                model = SAM3(
                    confidence_threshold=config.confidence_threshold,
                    resolution=config.resolution,
                    precision=config.precision,
                    device=resolved.device,
                    prompt_mode=prompt_mode,
                )
                logger.info("Using the Torch backend for SAM3")
                return TorchModelHandler(model, reference_batch)
            case _:
                logger.info("Model config didn't match any known type, falling back to a pass through model")
                return PassThroughModelHandler()
