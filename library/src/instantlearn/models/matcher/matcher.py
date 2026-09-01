# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Matcher model, based on the paper 'Segment Anything with One Shot Using All-Purpose Feature Matching'."""

import logging
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

from instantlearn.components.encoders import ImageEncoder
from instantlearn.components.feature_extractors import MaskedFeatureExtractor, ReferenceFeatures
from instantlearn.components.postprocessing import (
    PostProcessor,
    default_postprocessor,
)
from instantlearn.components.postprocessing.base import apply_postprocessing
from instantlearn.components.sam import SamDecoder, load_sam_model
from instantlearn.data.base.batch import Batch, Collatable
from instantlearn.data.base.prediction import Prediction
from instantlearn.data.base.sample import Sample
from instantlearn.device import DeviceInfo
from instantlearn.models._export_utils import (
    _INT4_MODES,
    IR_STEM,
    convert_and_save_openvino,
    export_onnx_graph,
    resolve_export_dir,
    write_metadata,
)
from instantlearn.models.model_card import ModelCard
from instantlearn.models.torch_adapter import CategoryRegistry, batch_to_tensors, dict_to_prediction
from instantlearn.models.torch_base import ExportConfig, TorchModel
from instantlearn.utils.constants import Backend, SAMModelName
from instantlearn.utils.errors import ModelNotFittedError

from ._card import _MATCHER_CARD
from .prompt_generators import BidirectionalPromptGenerator

logger = logging.getLogger(__name__)


class EncoderForwardFeaturesWrapper(nn.Module):
    """Wrapper for image encoder supporting both TIMM and HuggingFace backends for export.

    TIMM models expose a ``forward_features`` method that returns ``(B, N, D)`` directly.
    HuggingFace DINO models do not have ``forward_features``; instead they are called with
    ``pixel_values`` and return a structured output whose ``.last_hidden_state`` is ``(B, N, D)``.
    This wrapper detects which API is available at construction time and dispatches accordingly,
    so the same ONNX export path works regardless of which backend was used to train/fit the Matcher.
    """

    def __init__(
        self,
        encoder: nn.Module,
        ignore_token_length: int,
        input_size: int = 512,
    ) -> None:
        """Initialize the encoder wrapper.

        Args:
            encoder: The underlying encoder module (raw TIMM or HuggingFace model).
            ignore_token_length: Number of leading tokens to strip (CLS + register tokens).
            input_size: Input image size.
        """
        super().__init__()
        self.encoder = encoder
        self.ignore_token_length = ignore_token_length
        self.input_size = input_size
        self._is_timm = hasattr(encoder, "forward_features")
        self.register_buffer("IMAGENET_DEFAULT_MEAN", torch.tensor((0.485, 0.456, 0.406)))
        self.register_buffer("IMAGENET_DEFAULT_STD", torch.tensor((0.229, 0.224, 0.225)))

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Dispatch to the correct backend encoding API.

        Args:
            x: Pre-processed image tensor of shape ``(B, 3, H, W)``, float32, normalized.

        Returns:
            Raw patch token tensor of shape ``(B, N, D)`` including CLS / register tokens.
        """
        if self._is_timm:
            # TIMM: forward_features returns (B, N, D) directly.
            return self.encoder.forward_features(x)
        # HuggingFace: called with pixel_values; structured output exposes last_hidden_state.
        return self.encoder(pixel_values=x).last_hidden_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get normalized patch embeddings.

        Args:
            x: Image tensor of shape ``(B, 3, H, W)`` with values in ``[0, 255]``
                (any float/int dtype; it is cast to float before normalization).

        Returns:
            L2-normalized patch embeddings of shape ``(B, num_patches, embed_dim)``.
        """
        # Cast to float *before* deriving the ImageNet mean/std dtype, otherwise a
        # uint8 ``x`` would truncate mean/std to 0 (breaking normalization / div-by-zero).
        x = x.float() / 255.0
        imagenet_mean = self.IMAGENET_DEFAULT_MEAN.to(device=x.device, dtype=x.dtype)
        imagenet_std = self.IMAGENET_DEFAULT_STD.to(device=x.device, dtype=x.dtype)
        x = functional.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear")
        x = (x - imagenet_mean[None, :, None, None]) / imagenet_std[None, :, None, None]
        features = self._encode(x)
        features = features[:, self.ignore_token_length :, :]  # ignore CLS and other tokens
        return functional.normalize(features, p=2, dim=-1)


class MatcherInferenceGraph(nn.Module):
    """Traceable inference graph with frozen reference features for ONNX export."""

    def __init__(
        self,
        encoder: nn.Module,
        prompt_generator: BidirectionalPromptGenerator,
        sam_decoder: SamDecoder,
        ref_features: ReferenceFeatures,
        postprocessor: PostProcessor | None = None,
    ) -> None:
        """Initialize the inference graph with frozen reference features."""
        super().__init__()
        self.encoder = encoder
        self.prompt_generator = prompt_generator
        self.sam_decoder = sam_decoder

        # Register post-processor as a proper submodule
        # so parameters are captured during tracing/export.
        self.add_module("export_postprocessor", postprocessor)

        # Freeze reference features as model constants
        self.register_buffer("ref_embeddings", ref_features.ref_embeddings)
        self.register_buffer("masked_ref_embeddings", ref_features.masked_ref_embeddings)
        self.register_buffer("flatten_ref_masks", ref_features.flatten_ref_masks)
        self.register_buffer("category_ids", torch.tensor(ref_features.category_ids, device=ref_features.device))

    def forward(self, target_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single image forward pass for export: target_image [1, 3, H, W] → (masks, scores, labels)."""
        # Encode target [1, num_patches, embed_dim]
        target_embeddings = self.encoder(target_image)
        feature_device = target_embeddings.device

        # Align frozen reference tensors to target embedding device for trace-time safety.
        # This prevents mixed-device matmul when model buffers and encoder output diverge.
        ref_embeddings = self.ref_embeddings.to(feature_device)
        masked_ref_embeddings = self.masked_ref_embeddings.to(feature_device)
        flatten_ref_masks = self.flatten_ref_masks.to(feature_device)
        category_ids = self.category_ids.to(feature_device)

        # Spatial input is fixed to the encoder ``input_size`` (the OV IR is reshaped
        # to a static ``[1, 3, S, S]``), so ``original_sizes`` is deliberately the
        # traced input size; masks are rescaled to the true frame by the OV wrapper.
        height = torch.scalar_tensor(target_image.shape[2], dtype=torch.long, device=feature_device)
        width = torch.scalar_tensor(target_image.shape[3], dtype=torch.long, device=feature_device)
        original_sizes = torch.stack([height, width], dim=0).unsqueeze(0)

        # Generate prompts using frozen ref_features
        # point_prompts: [1, C, max_points, 4], num_points: [1, C], similarities: [1, C, feat_size, feat_size]
        point_prompts, similarities = self.prompt_generator.forward(
            ref_embeddings,
            masked_ref_embeddings,
            flatten_ref_masks,
            category_ids,
            target_embeddings,
            original_sizes,
        )

        # Decode using export-friendly method (single image, returns tensors)
        masks, scores, labels = self.sam_decoder.forward_export(
            target_image[0],  # Single image [3, H, W]
            category_ids,
            point_prompts[0],  # [C, max_points, 4]
            similarities[0],  # [C, feat_size, feat_size]
        )

        # Apply exportable post-processing (if any)
        if self.export_postprocessor is not None:
            masks, scores, labels = self.export_postprocessor(masks, scores, labels)

        return masks, scores, labels


class Matcher(TorchModel):
    """Matcher model for one-shot segmentation.

    Based on "[ICLR'24] Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching"
    https://arxiv.org/abs/2305.13310.

    The pipeline is fully traceable (ONNX/TorchScript compatible):
    - Encoder → MaskedFeatureExtractor → BidirectionalPromptGenerator → TraceableSamDecoder

    Examples:
        >>> from instantlearn.models import Matcher
        >>> from instantlearn.data.base import Batch
        >>> from instantlearn.data.base.sample import Category, Sample
        >>> import numpy as np

        >>> matcher = Matcher()

        >>> # Create reference sample (image is numpy HWC per the Sample contract)
        >>> ref_sample = Sample(
        ...     image=np.zeros((1024, 1024, 3), dtype=np.uint8),
        ...     masks=np.ones((1, 30, 30), dtype=bool),
        ...     is_reference=[True],
        ...     categories=[Category(1, "object")],
        ... )
        >>> ref_batch = Batch.collate([ref_sample])

        >>> # Create target sample
        >>> target_sample = Sample(
        ...     image=np.zeros((1024, 1024, 3), dtype=np.uint8),
        ...     is_reference=[False],
        ...     categories=[Category(0, "object")],
        ... )
        >>> target_batch = Batch.collate([target_sample])

        >>> # Run fit and predict
        >>> matcher.fit(ref_batch)
        >>> predictions = matcher.predict(target_batch)

        >>> isinstance(predictions, list)
        True

        >>> predictions[0].masks is not None
        True
    """

    def __init__(
        self,
        sam: SAMModelName = SAMModelName.SAM_HQ_TINY,
        num_foreground_points: int = 40,
        num_background_points: int = 2,
        encoder_model: str = "dinov3_large",
        confidence_threshold: float | None = 0.38,
        use_mask_refinement: bool = True,
        precision: str = "bf16",
        compile_models: bool = False,
        device: DeviceInfo | None = None,
        postprocessor: PostProcessor | None = None,
        similarity_threshold: float | None = None,
        num_grid_cells: int = 8,
        num_export_instances: int = 8,
    ) -> None:
        """Initialize the Matcher model.

        Args:
            sam: SAM model variant to use.
            num_foreground_points: Maximum foreground points per category.
            num_background_points: Background points per category.
            encoder_model: Image encoder model ID.
            confidence_threshold: Minimum confidence score for keeping predicted masks
                                 in the final output. Higher values = stricter filtering, fewer masks.
            use_mask_refinement: Whether to use 2-stage mask refinement with box prompts.
            precision: Model precision ("bf16", "fp32").
            compile_models: Whether to compile models with torch.compile.
            device: Physical device, or ``None`` to select automatically.
            postprocessor: Post-processor applied after predict().
                Defaults to :func:`~instantlearn.components.postprocessing.default_postprocessor`
                (MaskIoMNMS + BoxIoMNMS).
            similarity_threshold: When set, supplement bidirectional-matched points with
                additional target patches exceeding this similarity to the reference.
                Helps detect more objects when reference masks cover few patches.
                Set to None to disable. Default: None.
            num_grid_cells: Grid cells per dimension for spatial diversity filtering.
                When > 0, foreground points are deduplicated per grid cell before top-k
                selection, preventing point clustering on large objects. Default: 8.
            num_export_instances: Maximum instances per category the **exported**
                (ONNX/OpenVINO) model can detect. Each slot costs one SAM decoder pass,
                so latency scales linearly with this value. Only affects export; the
                PyTorch path is unbounded. Default: 8.
        """
        if postprocessor is None:
            postprocessor = default_postprocessor()
        super().__init__(device=device, precision=precision, postprocessor=postprocessor)
        # SAM predictor
        self.sam_predictor = load_sam_model(
            sam,
            device=self.device,
            precision=precision,
            compile_models=compile_models,
        )

        # Image encoder
        self.encoder = ImageEncoder(
            model_id=encoder_model,
            backend=Backend.HUGGINGFACE,
            device=self.device,
            precision=precision,
            compile_models=compile_models,
        )

        # Feature extractor
        self.masked_feature_extractor = MaskedFeatureExtractor(
            input_size=self.encoder.input_size,
            patch_size=self.encoder.patch_size,
            device=self.device,
        )

        # Prompt generator (includes filtering)
        self.prompt_generator = BidirectionalPromptGenerator(
            encoder_input_size=self.encoder.input_size,
            encoder_patch_size=self.encoder.patch_size,
            encoder_feature_size=self.encoder.feature_size,
            num_foreground_points=num_foreground_points,
            num_background_points=num_background_points,
            similarity_threshold=similarity_threshold,
            num_grid_cells=num_grid_cells,
        )

        # SAM decoder
        self.segmenter = SamDecoder(
            sam_predictor=self.sam_predictor,
            confidence_threshold=confidence_threshold,
            use_mask_refinement=use_mask_refinement,
            num_export_instances=num_export_instances,
            num_background_points=num_background_points,
        )

        # Reference features (set during fit)
        self.ref_features: ReferenceFeatures | None = None
        # Category identity (set during fit), used to build
        # ``Prediction.label_names`` at the numpy boundary.
        self.categories: CategoryRegistry = CategoryRegistry()

    @classmethod
    def card(cls) -> ModelCard:
        """Return the static capability descriptor for Matcher."""
        return _MATCHER_CARD

    @property
    def input_size(self) -> int:
        """Square input size expected by the encoder (e.g. 512)."""
        return self.encoder.input_size

    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Learn from reference images.

        Caches the reference features and category-name map on the instance.

        Args:
            reference: Reference data to learn from. Accepts:
                - Sample: A single reference sample
                - list[Sample]: A list of reference samples
                - Batch: A batch of reference samples
        """
        reference_batch = Batch.collate(reference)
        ref_embeddings = self.encoder(images=reference_batch.images)
        self.ref_features = self.masked_feature_extractor(
            ref_embeddings,
            reference_batch.masks,
            reference_batch.label_ids,
        )
        # Cache category identity so predict() can build Prediction.label_names.
        self.categories = CategoryRegistry.from_samples(reference_batch)

    def predict(self, target: Collatable) -> list[Prediction]:
        """Predict masks for target images.

        Args:
            target: Target data to infer. Accepts:
                - Sample: A single target sample
                - list[Sample]: A list of target samples
                - Batch: A batch of target samples
                - str | Path: A single image path
                - list[str] | list[Path]: Multiple image paths

        Returns:
            A list of ``Prediction`` objects, one per input image, with
            post-processing already applied.

        Raises:
            ModelNotFittedError: If ``fit()`` has not been called before ``predict()``.
        """
        target_batch = Batch.collate(target)
        if self.ref_features is None:
            msg = "Matcher requires fit() before predict(). Call model.fit(reference_sample) first."
            raise ModelNotFittedError(msg)

        # Convert inputs at the single torch boundary. ``Sample.image`` is numpy
        # HWC (read_image), so sizes must come from the tensor shape — calling
        # ``.size()`` on the raw numpy array would raise.
        # TODO: feed the encoder from these tensors too; the HF ``processor``
        # currently expects numpy HWC and does its own rescale/normalize, so
        # passing CHW float32 (0-255) here would double-transform.
        tensor_batch = batch_to_tensors(target_batch, device=str(self.ref_features.device))
        original_sizes = torch.tensor(
            [image.shape[-2:] for image in tensor_batch.images],
            device=self.ref_features.device,
        )

        # Encode all targets [T, num_patches, embed_dim]
        target_embeddings = self.encoder(images=target_batch.images)

        # Generate prompts [T, C, max_points, 4], [T, C], [T, C, feat_size, feat_size]
        point_prompts, similarities = self.prompt_generator(
            self.ref_features.ref_embeddings,
            self.ref_features.masked_ref_embeddings,
            self.ref_features.flatten_ref_masks,
            self.ref_features.category_ids,
            target_embeddings,
            original_sizes,
        )

        # Decode masks for all images. ``SamDecoder`` normalizes the images to CHW tensors on the SAM device/dtype
        # internally, so pass the converted ``tensor_batch`` (CHW) rather than the numpy ``Sample.image`` arrays.
        predictions = self.segmenter(
            tensor_batch.images,
            self.ref_features.category_ids,
            point_prompts=point_prompts,
            similarities=similarities,
        )
        predictions = apply_postprocessing(predictions, self.postprocessor)
        return [dict_to_prediction(pred, self.categories) for pred in predictions]

    @torch.no_grad()
    def _build_inference_graph(
        self,
        export_device: torch.device,
        *,
        sam_hq_tiny_fallback: bool,
    ) -> MatcherInferenceGraph:
        """Build the traceable inference graph with baked reference features.

        Args:
            export_device: Device to place the graph and reference tensors on.
            sam_hq_tiny_fallback: When ``True`` and the configured SAM variant is
                SAM-HQ-Tiny (non-deterministic under OpenVINO), fall back to
                SAM-HQ-base for the exported graph.

        Returns:
            An FP32 ``MatcherInferenceGraph`` ready for tracing.

        Raises:
            ModelNotFittedError: If ``fit()`` has not been called.
        """
        if self.ref_features is None:
            msg = "No reference features. Call fit() first."
            raise ModelNotFittedError(msg)

        fallback_segmenter = None
        if sam_hq_tiny_fallback and self.sam_predictor.sam_model_name == SAMModelName.SAM_HQ_TINY:
            logger.warning(
                "SAM-HQ-Tiny is not supported for OpenVINO export. "
                "Some of the layers are non-deterministic and so the exported model is not reliable for inference. "
                "Falling back to SAM-HQ-base for the exported model. "
                "SAM-HQ-base weights will be downloaded if not already cached.",
            )
            fallback_predictor = load_sam_model(
                SAMModelName.SAM_HQ_BASE,
                device="cpu",
                precision="fp32",
            )
            fallback_segmenter = SamDecoder(
                sam_predictor=fallback_predictor,
                confidence_threshold=self.segmenter.confidence_threshold,
                use_mask_refinement=self.segmenter.use_mask_refinement,
                num_export_instances=self.segmenter.num_export_instances,
                num_background_points=self.segmenter.num_background_points,
            )

        self.sam_predictor.sync_device(export_device, dtype=torch.float32)
        self.segmenter.device = self.sam_predictor.device
        ref_features = self.ref_features.to(export_device)
        export_decoder = fallback_segmenter if fallback_segmenter is not None else self.segmenter

        # Force FP32 for stable CPU tracing.
        return (
            MatcherInferenceGraph(
                encoder=EncoderForwardFeaturesWrapper(
                    self.encoder._model.model,  # noqa: SLF001
                    ignore_token_length=self.encoder._model.ignore_token_length,  # noqa: SLF001
                ),
                prompt_generator=self.prompt_generator,
                sam_decoder=export_decoder,
                ref_features=ref_features,
                postprocessor=self.postprocessor,
            )
            .to(export_device)
            .float()
        )

    @torch.no_grad()
    def to_onnx(self, export_path: str | Path | None = None, config: ExportConfig | None = None) -> Path:
        """Export the baked Matcher graph to ONNX.

        Requires ``fit()`` first — the reference features are baked into the
        exported graph as constants.

        Args:
            export_path: Destination directory. ``None`` writes to a temporary
                directory that is *not* auto-deleted (so the returned path stays
                valid).
            config: Export options. ``None`` uses :class:`ExportConfig` defaults.

        Returns:
            Path to the exported ``model.onnx`` file.

        Raises:
            ModelNotFittedError: If ``fit()`` has not been called.
        """
        config = config or ExportConfig()
        export_dir = resolve_export_dir(export_path, self.card().family)

        # Prefer the encoder's own device for the torch-native ONNX graph.
        export_device = self.ref_features.device  # type: ignore[union-attr]
        first_encoder_param = next(iter(self.encoder._model.model.parameters()), None)  # noqa: SLF001
        if isinstance(first_encoder_param, torch.Tensor):
            export_device = first_encoder_param.device

        graph = self._build_inference_graph(export_device, sam_hq_tiny_fallback=False)
        onnx_path = export_dir / f"{IR_STEM}.onnx"
        export_onnx_graph(
            graph,
            onnx_path,
            export_device,
            self.encoder.input_size,
            dynamic_shapes=config.dynamic_shapes,
            opset=config.opset,
        )
        return onnx_path

    @torch.no_grad()
    def to_openvino(self, export_path: str | Path | None = None, config: ExportConfig | None = None) -> Path:
        """Export the baked Matcher graph to an OpenVINO IR directory.

        Conversion goes Torch -> ONNX -> OpenVINO IR (ONNX has much better
        operator coverage than a direct Torch->OV conversion). Requires
        ``fit()`` first — reference features are baked into the graph. Also
        writes a ``metadata.json`` so :class:`MatcherOpenVINO` can build
        ``Prediction.label_names`` without re-fitting.

        Args:
            export_path: Destination directory for the IR. ``None`` writes to a
                temporary directory that is *not* auto-deleted.
            config: Export options (compression, opset, ...). ``None`` uses
                :class:`ExportConfig` defaults.

        Returns:
            Path to the exported IR **directory** (containing ``model.xml`` /
            ``model.bin`` / ``metadata.json``).

        Raises:
            ImportError: If OpenVINO is not installed.
            ModelNotFittedError: If ``fit()`` has not been called.
            ValueError: If an INT4 compression mode is requested.
        """
        config = config or ExportConfig()
        if config.compression in _INT4_MODES:
            msg = (
                "INT4 compressed models for Matcher produce random noisy masks and are not accurate. "
                "Please use INT8 compression or no compression for Matcher exports."
            )
            raise ValueError(msg)

        try:
            import openvino  # noqa: F401, PLC0415
        except ImportError as e:
            msg = "OpenVINO is not installed. Please install it to use OpenVINO export."
            raise ImportError(msg) from e

        export_dir = resolve_export_dir(export_path, self.card().family)
        export_device = torch.device("cpu")
        graph = self._build_inference_graph(export_device, sam_hq_tiny_fallback=True)

        # Keep the OpenVINO intermediate ONNX static: the baked graph is reshaped
        # to a static input during conversion, and dynamic axes can cause
        # infer-time mismatch.
        onnx_path = export_dir / f"{IR_STEM}.onnx"
        export_onnx_graph(
            graph,
            onnx_path,
            export_device,
            self.encoder.input_size,
            dynamic_shapes=False,
            opset=config.opset,
        )

        convert_and_save_openvino(
            graph,
            onnx_path,
            export_device,
            export_dir,
            self.encoder.input_size,
            compression=config.compression,
            keep_intermediate=config.keep_intermediate,
        )

        write_metadata(export_dir, self.encoder.input_size, self.encoder.patch_size, self.categories.id_to_name)
        return export_dir
