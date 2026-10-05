#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

"""Factory building ready-to-use models for the inference pipeline."""

import logging
import os
import re
import shutil
import tempfile
from pathlib import Path

from instantlearn.data.base.batch import Batch
from instantlearn.device import DeviceInfo, ResolvedDevice, resolve_device_for_model
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
from instantlearn.models.model_card import ModelCard
from instantlearn.models.sam3.sam3 import MODEL_NAMES as SAM3_MODEL_NAMES
from instantlearn.models.sam3.sam3 import SAM3_MODEL_ID
from instantlearn.models.torch_base import ExportConfig
from instantlearn.utils.constants import Backend, CompressionMode

from domain.services.schemas.processor import (
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

# Written into an IR directory only after every sub-model has been verified, so a
# SAM3 cache entry is usable if and only if this file is present.
_IR_COMPLETE_MARKER = ".instantlearn-complete"

# Presence of this file lets SAM3OpenVINO load the tokenizer without network access.
_TOKENIZER_MARKER_FILE = "tokenizer.json"


class ModelFactory:
    def __init__(self, device_service: DeviceService) -> None:
        self._device_service = device_service

    def _resolve_device(
        self,
        model_card: ModelCard,
        configured_device: str | None,
        allowed_runtimes: tuple[Backend, ...] = (Backend.OPENVINO, Backend.TORCH),
    ) -> ResolvedDevice:
        """Resolve a configured preference to a concrete model runtime route."""
        device_str = configured_device or "auto"
        preferred_device = self._device_service.resolve_preference(device_str)
        resolved = resolve_device_for_model(
            model_card=model_card,
            device=preferred_device,
            devices=self._device_service.list_devices(),
            allowed_runtimes=allowed_runtimes,
            allow_fallback=True,
        )
        if device_str != "auto" and preferred_device is None:
            resolved = ResolvedDevice(
                device=resolved.device,
                runtime=resolved.runtime,
                runtime_id=resolved.runtime_id,
                fallback_used=True,
            )
        elif resolved.fallback_used:
            logger.warning(
                "Device %r is not supported by model %s; using %s on %s.",
                device_str,
                model_card.name,
                resolved.runtime.value,
                resolved.device.key,
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

    def create(
        self, reference_batch: Batch | None, config: ModelConfig | None, configured_device: str | None = None
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

        model_card = _model_card(config)
        if model_card is None:
            logger.error("Model config didn't match any known type, falling back to pass-through processing")
            return PassThroughModelHandler()

        resolved_device = self._resolve_device(model_card, configured_device)
        model = self._create_model(
            config=config,
            reference_batch=reference_batch,
            model_card=model_card,
            resolved_device=resolved_device,
        )
        if model is None:
            logger.info("Model config didn't match any known type, falling back to a pass through processing")
            return PassThroughModelHandler()

        return InferenceModelHandler(model)

    def _create_model(
        self,
        config: ModelConfig,
        reference_batch: Batch,
        model_card: ModelCard,
        resolved_device: ResolvedDevice,
    ) -> Model | None:
        """Construct and fit a model, converting it to OpenVINO when enabled.

        OpenVINO routes build and export the temporary Torch model on CPU because
        the selected OpenVINO target may not support Torch. The device-agnostic IR
        is then compiled and used for inference on the selected target device.
        """
        use_openvino = resolved_device.runtime == Backend.OPENVINO
        if isinstance(config, Sam3Config):
            return self._create_sam3(
                config=config,
                reference_batch=reference_batch,
                resolved_device=resolved_device,
            )

        torch_device = resolved_device.device
        if use_openvino:
            # The final OpenVINO target may not support Torch. Build and export the
            # temporary Torch model on CPU, then compile the IR on the selected target.
            torch_device = self._resolve_device(
                model_card,
                "cpu",
                allowed_runtimes=(Backend.TORCH,),
            ).device
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
                    device=torch_device,
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
                    device=torch_device,
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
                    num_grid_cells=config.num_grid_cells,
                    precision=precision,
                    device=torch_device,
                )
            case _:
                return None

        if use_openvino:
            ov_model = self._export_and_load(
                model=model,
                ov_cls=ov_cls,
                reference_batch=reference_batch,
                target_device=resolved_device.device,
                compression=config.ov_compression,
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
        target_device: DeviceInfo,
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
            target_device: Physical device that will compile and run the OpenVINO model.
            compression: Weight compression mode for the exported IR.

        Returns:
            The compiled OpenVINO model.
        """
        logger.info("Using the OpenVINO backend for %s", type(model).__name__)

        model.fit(reference_batch)

        # Export on CPU: tracing on XPU/CUDA hits device-specific issues, the resulting
        # IR is device-agnostic, and the target device is applied at compile time below.
        model.cpu()
        with tempfile.TemporaryDirectory(prefix="instantlearn-ir-") as tmp_dir:
            logger.info("Exporting %s to OpenVINO IR (compression=%s)", type(model).__name__, compression.value)
            ir_dir = model.to_openvino(tmp_dir, ExportConfig(compression=compression))
            logger.info("Loading %s from %s on %s", ov_cls.__name__, ir_dir, target_device.key)
            return ov_cls(model_dir=ir_dir, device=target_device)

    def _create_sam3(
        self,
        config: Sam3Config,
        reference_batch: Batch,
        resolved_device: ResolvedDevice,
    ) -> Model:
        """Build a SAM3 model, reusing a cached OpenVINO IR when available."""
        logger.info("Initializing a SAM3 instance")
        has_bboxes = any(s.bboxes is not None for s in reference_batch.samples)
        prompt_mode = Sam3PromptMode.CANVAS if has_bboxes else Sam3PromptMode.CLASSIC

        if resolved_device.runtime == Backend.OPENVINO:
            logger.info("Using the OpenVINO backend for SAM3")
            export_device = self._resolve_device(SAM3.card(), "cpu", (Backend.TORCH,)).device
            ir_dir = self._resolve_sam3_ir(
                config=config,
                compression=config.ov_compression,
                export_device=export_device,
            )
            model: Model = SAM3OpenVINO(
                model_dir=ir_dir,
                device=resolved_device.device,
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
                device=resolved_device.device,
                prompt_mode=prompt_mode,
            )

        model.fit(reference_batch)
        return model

    @staticmethod
    def _resolve_sam3_ir(
        config: Sam3Config,
        compression: CompressionMode,
        export_device: DeviceInfo,
    ) -> Path:
        """Return a SAM3 OpenVINO IR directory, exporting it only on a cache miss.

        Unlike the matcher family, the SAM3 export contains no reference data:
        the sub-models are traced straight from the pretrained weights and the
        prompts are applied at ``fit()`` time. The IR therefore survives prompt
        changes and is cached on disk, keyed by model id, resolution, and
        compression mode.

        The export runs inside a staging directory next to the cache entry and
        is moved into place only once it is complete. This keeps the published
        cache entry atomic (an interrupted export can never be mistaken for a
        usable one) and guarantees the library's intermediate artefacts
        (``onnx/``, ``openvino-<mode>-fp16-source/``) are removed even when the
        export raises.

        Args:
            config: SAM3 configuration providing the input resolution.
            compression: Weight compression mode for the exported IR.
            export_device: Torch-compatible physical device used for export.

        Returns:
            Path to a complete SAM3 OpenVINO IR directory.
        """
        settings = get_settings()
        cache_root = Path(settings.ir_cache_dir)
        export_root = cache_root / f"sam3-{_slugify(SAM3_MODEL_ID)}-r{config.resolution}"
        ir_dir = export_root / f"openvino-{compression.value}"

        if _sam3_ir_complete(ir_dir):
            logger.info("Reusing cached SAM3 OpenVINO IR: %s", ir_dir)
            return ir_dir

        if ir_dir.exists():
            logger.info("Discarding incomplete SAM3 IR cache entry at %s", ir_dir)
            shutil.rmtree(ir_dir, ignore_errors=True)

        logger.info("No cached SAM3 IR at %s, exporting it now (this may take several minutes)", ir_dir)
        export_root.mkdir(parents=True, exist_ok=True)
        # Stage inside the cache root so publishing is a same-filesystem rename.
        staging_root = Path(tempfile.mkdtemp(prefix=f"{export_root.name}-{compression.value}-", dir=cache_root))
        try:
            exported_dir = ModelFactory._export_sam3_ir(
                config=config,
                compression=compression,
                export_root=staging_root,
                export_device=export_device,
            )
            _mark_sam3_ir_complete(exported_dir)
            _publish_ir_dir(exported_dir, ir_dir)
        finally:
            # Removes the staging tree wholesale: the ONNX dump and the fp16 compression
            # source the library leaves behind on failure, plus the moved IR on success.
            shutil.rmtree(staging_root, ignore_errors=True)

        logger.info("SAM3 OpenVINO IR cached at %s", ir_dir)
        return ir_dir

    @staticmethod
    def _export_sam3_ir(
        config: Sam3Config,
        compression: CompressionMode,
        export_root: Path,
        export_device: DeviceInfo,
    ) -> Path:
        """Export a throwaway torch SAM3 to OpenVINO IR under *export_root*."""
        exporter = SAM3(
            # This instance only exports, it never predicts: CPU keeps tracing stable
            # and leaves accelerator memory free for the compiled OpenVINO model.
            device=export_device,
            precision=_OPENVINO_PRECISION,
            resolution=config.resolution,
            model_id=SAM3_MODEL_ID,
        )
        try:
            return exporter.to_openvino(export_root, ExportConfig(compression=compression))
        finally:
            del exporter
            empty_accelerator_cache()


def _model_card(config: ModelConfig) -> ModelCard | None:
    """Return the capability descriptor for a supported model config."""
    match config:
        case MatcherConfig():
            return Matcher.card()
        case PerDinoConfig():
            return PerDino.card()
        case SoftMatcherConfig():
            return SoftMatcher.card()
        case Sam3Config():
            return SAM3.card()
        case _:
            return None


def _slugify(value: str) -> str:
    """Return a filesystem-safe version of *value*."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def _sam3_ir_complete(ir_dir: Path) -> bool:
    """Return ``True`` when *ir_dir* is a fully published SAM3 IR cache entry.

    Requires the completion marker written by :func:`_mark_sam3_ir_complete` in
    addition to a non-empty ``.xml``/``.bin`` pair per sub-model. The marker is
    what distinguishes a finished export from one that was interrupted midway:
    a truncated file is still non-empty, so size checks alone cannot detect it.
    Cache entries created before the marker existed are re-exported once.
    """
    if not ir_dir.is_dir():
        return False
    marker = ir_dir / _IR_COMPLETE_MARKER
    if not marker.exists():
        logger.debug("SAM3 IR cache miss: %s has no completion marker", ir_dir)
        return False
    for name in SAM3_MODEL_NAMES:
        for suffix in (".xml", ".bin"):
            path = ir_dir / f"{name}{suffix}"
            if not path.exists() or path.stat().st_size == 0:
                logger.debug("SAM3 IR cache miss: %s is missing or empty", path)
                return False
    return True


def _mark_sam3_ir_complete(ir_dir: Path) -> None:
    """Validate *ir_dir* and stamp it with the completion marker.

    Called on the staged directory just before it is published, so the marker
    travels with the atomic rename and is never visible on a partial export.

    Raises:
        FileNotFoundError: If a sub-model IR file is missing or empty.
    """
    missing = [
        f"{name}{suffix}"
        for name in SAM3_MODEL_NAMES
        for suffix in (".xml", ".bin")
        if not (ir_dir / f"{name}{suffix}").exists() or (ir_dir / f"{name}{suffix}").stat().st_size == 0
    ]
    if missing:
        msg = f"Incomplete SAM3 OpenVINO export in {ir_dir}: missing or empty {', '.join(missing)}"
        raise FileNotFoundError(msg)

    # Not fatal: SAM3OpenVINO falls back to the HuggingFace hub, which only fails offline.
    if not (ir_dir / _TOKENIZER_MARKER_FILE).exists():
        logger.warning(
            "SAM3 IR at %s has no %s; the tokenizer will be fetched from the hub at load time",
            ir_dir,
            _TOKENIZER_MARKER_FILE,
        )

    (ir_dir / _IR_COMPLETE_MARKER).touch()


def _publish_ir_dir(staged_dir: Path, target_dir: Path) -> None:
    """Move *staged_dir* onto *target_dir* as a single rename.

    The staged directory lives in the same cache root as the target, so the
    rename is atomic and readers never observe a half-written cache entry.
    Normalising onto *target_dir* also keeps the on-disk layout owned by this
    factory even if the library changes its own export directory naming.
    """
    if staged_dir.name != target_dir.name:
        logger.debug("Publishing exported IR %s under the cache name %s", staged_dir.name, target_dir.name)

    if target_dir.exists():
        if _sam3_ir_complete(target_dir):
            # A concurrent export won the race; its entry is complete, so keep it.
            logger.info("SAM3 IR was published concurrently at %s, discarding the staged copy", target_dir)
            return
        logger.info("Replacing incomplete SAM3 IR cache entry at %s", target_dir)
        shutil.rmtree(target_dir, ignore_errors=True)

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged_dir, target_dir)
