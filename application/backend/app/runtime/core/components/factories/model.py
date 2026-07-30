#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

"""Factory building ready-to-use models for the inference pipeline.

The factory owns every backend decision. It constructs the PyTorch model, fits
it with the reference batch and — when OpenVINO is enabled — exports it and
loads the corresponding OpenVINO sibling. The resulting model is handed to a
single :class:`~runtime.core.components.models.inference_model.InferenceModelHandler`,
so the pipeline itself stays backend-agnostic.
"""

import logging
import re
import shutil
import tempfile
from pathlib import Path

from instantlearn.data.base.batch import Batch
from instantlearn.models import (
    SAM3,
    Matcher,
    MatcherOpenVINO,
    Model,
    OpenVINOModel,
    PerDino,
    PerDinoOpenVINO,
    SAM3OpenVINO,
    Sam3PromptMode,
    SoftMatcher,
    SoftMatcherOpenVINO,
    TorchModel,
)
from instantlearn.models.sam3.sam3 import MODEL_NAMES as SAM3_MODEL_NAMES
from instantlearn.models.sam3.sam3 import SAM3_LIBRARY_MODEL_ID
from instantlearn.models.torch_base import ExportConfig
from instantlearn.utils.constants import CompressionMode

from domain.services.schemas.device import DeviceInfo, DeviceType
from domain.services.schemas.processor import (
    CompressionPreset,
    MatcherConfig,
    ModelConfig,
    PerDinoConfig,
    Sam3Config,
    SoftMatcherConfig,
)
from runtime.core.components.base import ModelHandler
from runtime.core.components.models.inference_model import InferenceModelHandler, empty_accelerator_cache
from runtime.core.components.models.passthrough_model import PassThroughModelHandler
from runtime.services.device import DeviceService
from settings import get_settings

logger = logging.getLogger(__name__)

# OpenVINO graphs are traced in fp32; low-precision weights come from the
# compression pass driven by ``ExportConfig.compression`` instead.
_OPENVINO_PRECISION = "fp32"


class ModelFactory:
    def __init__(
        self,
        device_service: DeviceService,
    ) -> None:
        self._device_service = device_service

    def _resolve_device(self, configured_device: str | None) -> DeviceInfo:
        """Resolve a configured device string into a concrete :class:`DeviceInfo`."""
        device_info = self._device_service.resolve(configured_device or "auto")
        if device_info.type == DeviceType.AUTO:
            device_info = self._device_service.resolve_auto()
        logger.info(
            "Accelerator selected: torch=%s ov=%s (configured=%r)",
            device_info.as_torch,
            device_info.as_openvino,
            configured_device,
        )
        return device_info

    def create(
        self,
        reference_batch: Batch | None,
        config: ModelConfig | None,
        configured_device: str | None = None,
    ) -> ModelHandler:
        """Build a fully initialised model handler for the given configuration.

        Args:
            reference_batch: Reference prompts. ``None`` yields a pass-through handler.
            config: Model configuration. ``None`` yields a pass-through handler.
            configured_device: User-selected device string, or ``None`` for auto.

        Returns:
            An :class:`InferenceModelHandler` wrapping a fitted model, or a
            :class:`PassThroughModelHandler` when inference cannot or should not run.
        """
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

        device_info = self._resolve_device(configured_device)
        model = self._create_model(
            config=config,
            reference_batch=reference_batch,
            device_info=device_info,
            use_openvino=settings.processor_openvino_enabled,
        )
        if model is None:
            logger.info("Model config didn't match any known type, falling back to a pass through processing")
            return PassThroughModelHandler()

        return InferenceModelHandler(model)

    def _create_model(
        self,
        config: ModelConfig,
        reference_batch: Batch,
        device_info: DeviceInfo,
        *,
        use_openvino: bool,
    ) -> Model | None:
        """Construct and fit a model, converting it to OpenVINO when enabled."""
        selected_device = device_info.as_torch
        # If the model is converted to the OV format the precision must be fp32:
        # the graph is traced in full precision and compressed afterwards.
        precision = _OPENVINO_PRECISION if use_openvino else config.precision

        match config:
            case MatcherConfig() as config:
                logger.info("Initializing a Matcher instance")
                ov_cls: type[OpenVINOModel] = MatcherOpenVINO
                model: TorchModel = Matcher(
                    sam=config.sam_model,
                    encoder_model=config.encoder_model,
                    num_foreground_points=config.num_foreground_points,
                    num_background_points=config.num_background_points,
                    confidence_threshold=config.confidence_threshold,
                    use_mask_refinement=config.use_mask_refinement,
                    similarity_threshold=config.similarity_threshold,
                    num_grid_cells=config.num_grid_cells,
                    precision=precision,
                    device=selected_device,
                )
            case PerDinoConfig() as config:
                logger.info("Initializing a PerDINO instance")
                ov_cls = PerDinoOpenVINO
                model = PerDino(
                    sam=config.sam_model,
                    encoder_model=config.encoder_model,
                    num_foreground_points=config.num_foreground_points,
                    num_background_points=config.num_background_points,
                    num_grid_cells=config.num_grid_cells,
                    point_selection_threshold=config.point_selection_threshold,
                    confidence_threshold=config.confidence_threshold,
                    precision=precision,
                    device=selected_device,
                )
            case SoftMatcherConfig() as config:
                logger.info("Initializing a SoftMatcher instance")
                ov_cls = SoftMatcherOpenVINO
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
                    precision=precision,
                    device=selected_device,
                )
            case Sam3Config() as config:
                return self._create_sam3(
                    config=config,
                    reference_batch=reference_batch,
                    device_info=device_info,
                    use_openvino=use_openvino,
                )
            case _:
                return None

        if use_openvino:
            ov_model = self._export_and_load(
                model=model,
                ov_cls=ov_cls,
                reference_batch=reference_batch,
                ov_device=device_info.as_openvino,
                compression=_compression_mode(config),
            )
            del model  # release the torch graph as soon as the IR is compiled
            empty_accelerator_cache()
            return ov_model

        logger.info("Using the Torch backend for %s", type(model).__name__)
        model.fit(reference_batch)
        return model

    @staticmethod
    def _export_and_load(
        model: TorchModel,
        ov_cls: type[OpenVINOModel],
        reference_batch: Batch,
        ov_device: str,
        compression: CompressionMode,
    ) -> Model:
        """Fit, export to OpenVINO IR, and load the matching OpenVINO sibling.

        Matcher, SoftMatcher, and PerDino bake the reference features into the
        exported graph, so the IR is only valid for this exact reference batch
        and is written to a temporary directory. The OpenVINO sibling reads the
        IR fully while compiling, so the directory can be discarded right after.

        Args:
            model: Unfitted torch model.
            ov_cls: OpenVINO sibling class that loads the exported IR.
            reference_batch: Reference prompts to bake into the graph.
            ov_device: OpenVINO device string, e.g. ``"CPU"`` or ``"GPU.0"``.
            compression: Weight compression mode for the exported IR.

        Returns:
            The compiled OpenVINO model.
        """
        logger.info("Using the OpenVINO backend for %s", type(model).__name__)

        model.fit(reference_batch)

        # Export on CPU to avoid XPU/CUDA compilation issues during tracing.
        # The exported model can then run on any OpenVINO device.
        model.cpu()  #todo for sure?
        with tempfile.TemporaryDirectory(prefix="instantlearn-ir-") as tmp_dir:
            logger.info("Exporting %s to OpenVINO IR (compression=%s)", type(model).__name__, compression.value)
            ir_dir = model.to_openvino(tmp_dir, ExportConfig(compression=compression))
            logger.info("Loading %s from %s on %s", ov_cls.__name__, ir_dir, ov_device)
            return ov_cls(model_dir=ir_dir, device=ov_device)

    def _create_sam3(
        self,
        config: Sam3Config,
        reference_batch: Batch,
        device_info: DeviceInfo,
        *,
        use_openvino: bool,
    ) -> Model:
        """Build a SAM3 model, reusing a cached OpenVINO IR when available."""
        logger.info("Initializing a SAM3 instance")
        has_bboxes = any(s.bboxes is not None for s in reference_batch.samples)
        prompt_mode = Sam3PromptMode.CANVAS if has_bboxes else Sam3PromptMode.CLASSIC

        if use_openvino:
            logger.info("Using the OpenVINO backend for SAM3")
            ir_dir = self._resolve_sam3_ir(config=config, compression=_compression_mode(config))
            model: Model = SAM3OpenVINO(
                ir_path=ir_dir,
                device=device_info.as_openvino,
                confidence_threshold=config.confidence_threshold,
                resolution=config.resolution,
                prompt_mode=prompt_mode,
            )
        else:
            logger.info("Using the Torch backend for SAM3")
            model = SAM3(
                confidence_threshold=config.confidence_threshold,
                resolution=config.resolution,
                precision=config.precision,
                device=device_info.as_torch,
                prompt_mode=prompt_mode,
            )

        model.fit(reference_batch)
        return model

    @staticmethod
    def _resolve_sam3_ir(config: Sam3Config, compression: CompressionMode) -> Path:
        """Return a SAM3 OpenVINO IR directory, exporting it only on a cache miss.

        Unlike the matcher family, the SAM3 export contains no reference data:
        the sub-models are traced straight from the pretrained weights and the
        prompts are applied at ``fit()`` time. The IR therefore survives prompt
        changes and is cached on disk, keyed by model id, resolution, and
        compression mode.

        Args:
            config: SAM3 configuration providing the input resolution.
            compression: Weight compression mode for the exported IR.

        Returns:
            Path to a complete SAM3 OpenVINO IR directory.
        """
        settings = get_settings()
        export_root = Path(settings.ir_cache_dir) / f"sam3-{_slugify(SAM3_LIBRARY_MODEL_ID)}-r{config.resolution}"
        ir_dir = export_root / f"openvino-{compression.value}"

        if _sam3_ir_complete(ir_dir):
            logger.info("Reusing cached SAM3 OpenVINO IR: %s", ir_dir)
            return ir_dir

        logger.info("No cached SAM3 IR at %s, exporting it now (this may take several minutes)", ir_dir)
        export_root.mkdir(parents=True, exist_ok=True)
        exporter = SAM3(
            device="cpu",  #todo for sure?
            precision=_OPENVINO_PRECISION,
            resolution=config.resolution,
            model_id=SAM3_LIBRARY_MODEL_ID,
        )
        try:
            exported_dir = exporter.to_openvino(export_root, ExportConfig(compression=compression))
        except Exception:
            shutil.rmtree(ir_dir, ignore_errors=True)  # cleanup corrupted directory
            raise
        finally:
            del exporter
            empty_accelerator_cache()

        logger.info("SAM3 OpenVINO IR cached at %s", exported)
        return exported_dir


def _compression_mode(config: ModelConfig) -> CompressionMode:
    """Return the weight compression mode for *config*.

    Only Matcher currently exposes a user-facing preset; every other model
    falls back to the throughput preset.
    """
    preset: CompressionPreset = getattr(config, "preset", CompressionPreset.THROUGHPUT)
    return preset.to_compression_mode()


def _slugify(value: str) -> str:
    """Return a filesystem-safe version of *value*."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def _sam3_ir_complete(ir_dir: Path) -> bool:
    """Return ``True`` when *ir_dir* holds every SAM3 sub-model as a non-empty IR pair."""
    if not ir_dir.is_dir():
        return False
    for name in SAM3_MODEL_NAMES:
        for suffix in (".xml", ".bin"):
            path = ir_dir / f"{name}{suffix}"
            if not path.exists() or path.stat().st_size == 0:
                logger.debug("SAM3 IR cache miss: %s is missing or empty", path)
                return False
    return True
