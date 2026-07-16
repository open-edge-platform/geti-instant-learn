# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PerDino model."""

import logging
from pathlib import Path

import torch
from torch import nn

from instantlearn.components import CosineSimilarity, SamDecoder
from instantlearn.components.encoders import ImageEncoder
from instantlearn.components.feature_extractors import MaskedFeatureExtractor, ReferenceFeatures
from instantlearn.components.postprocessing import (
    PostProcessor,
    default_postprocessor,
)
from instantlearn.components.postprocessing.base import apply_postprocessing
from instantlearn.components.sam import load_sam_model
from instantlearn.data.base.batch import Batch, Collatable
from instantlearn.data.base.prediction import Prediction
from instantlearn.data.base.sample import Sample
from instantlearn.models._export_utils import (
    _INT4_MODES,
    IR_STEM,
    convert_and_save_openvino,
    export_onnx_graph,
    resolve_export_dir,
    write_metadata,
)
from instantlearn.models.matcher import EncoderForwardFeaturesWrapper
from instantlearn.models.model_card import ModelCard
from instantlearn.models.torch_adapter import batch_to_tensors, dict_to_prediction
from instantlearn.models.torch_base import ExportConfig, TorchModel
from instantlearn.utils.constants import Backend, SAMModelName
from instantlearn.utils.errors import ModelNotFittedError

from ._card import _PERDINO_CARD
from .prompt_generators import GridPromptGenerator

logger = logging.getLogger(__name__)


class PerDinoInferenceGraph(nn.Module):
    """Traceable PerDino inference graph with frozen reference features for ONNX export.

    Bakes the averaged masked reference embeddings and category ids as buffers so
    the exported graph takes a single ``target_image`` and returns
    ``(masks, scores, labels)``. The pipeline is
    encoder -> CosineSimilarity -> GridPromptGenerator (export path) -> SAM decoder.
    """

    def __init__(
        self,
        encoder: nn.Module,
        similarity_matcher: CosineSimilarity,
        prompt_generator: GridPromptGenerator,
        sam_decoder: SamDecoder,
        ref_features: ReferenceFeatures,
        postprocessor: PostProcessor | None = None,
    ) -> None:
        """Initialize the inference graph with frozen reference features."""
        super().__init__()
        self.encoder = encoder
        self.similarity_matcher = similarity_matcher
        self.prompt_generator = prompt_generator
        self.sam_decoder = sam_decoder

        # Register post-processor as a proper submodule so its parameters are
        # captured during tracing/export.
        self.add_module("export_postprocessor", postprocessor)

        # PerDino only needs the averaged masked reference embedding (per
        # category) to build the cosine-similarity maps; the prompt generator
        # then works purely off those maps.
        self.register_buffer("masked_ref_embeddings", ref_features.masked_ref_embeddings)
        self.register_buffer("category_ids", torch.tensor(ref_features.category_ids, device=ref_features.device))

    def forward(self, target_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single image forward pass for export: target_image [1, 3, H, W] -> (masks, scores, labels)."""
        # Encode target [1, num_patches, embed_dim]
        target_embeddings = self.encoder(target_image)
        feature_device = target_embeddings.device

        # Align frozen reference tensors to the target embedding device for
        # trace-time safety (avoids mixed-device matmul if buffers diverge).
        masked_ref_embeddings = self.masked_ref_embeddings.to(feature_device)
        category_ids = self.category_ids.to(feature_device)

        # Spatial input is fixed to the encoder ``input_size`` (the OV IR is reshaped
        # to a static ``[1, 3, S, S]``), so ``original_sizes`` is deliberately the
        # traced input size; masks are rescaled to the true frame by the OV wrapper.
        height = torch.scalar_tensor(target_image.shape[2], dtype=torch.long, device=feature_device)
        width = torch.scalar_tensor(target_image.shape[3], dtype=torch.long, device=feature_device)
        original_sizes = torch.stack([height, width], dim=0).unsqueeze(0)

        # Cosine similarity maps [1, C, feat_size, feat_size]
        similarities = self.similarity_matcher(masked_ref_embeddings, target_embeddings, category_ids)

        # Grid prompts (export path) [1, C, max_points, 4]
        point_prompts = self.prompt_generator(similarities, category_ids, original_sizes)

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


class PerDino(TorchModel):
    """PerDino algorithm model for one-shot segmentation.

    Matches reference objects to target images by comparing features extracted by DINOv2
    using cosine similarity. A grid prompt generator creates multi-object aware prompts.

    The pipeline is fully traceable (ONNX/TorchScript compatible):
    - Encoder → MaskedFeatureExtractor → CosineSimilarity → GridPromptGenerator → SamDecoder

    Examples:
        >>> from instantlearn.models import PerDino
        >>> from instantlearn.data.base import Batch
        >>> from instantlearn.data.base.sample import Category, Sample
        >>> import numpy as np

        >>> perdino = PerDino()

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
        >>> perdino.fit(ref_batch)
        >>> predictions = perdino.predict(target_batch)

        >>> isinstance(predictions, list)
        True
    """

    def __init__(
        self,
        sam: SAMModelName = SAMModelName.SAM_HQ_TINY,
        encoder_model: str = "dinov3_large",
        num_foreground_points: int = 40,
        num_background_points: int = 2,
        num_grid_cells: int = 16,
        point_selection_threshold: float = 0.65,
        confidence_threshold: float | None = 0.42,
        precision: str = "bf16",
        compile_models: bool = False,
        device: str = "cuda",
        postprocessor: PostProcessor | None = None,
    ) -> None:
        """Initialize the PerDino model.

        Args:
            sam: SAM model variant to use.
            encoder_model: ImageEncoder model ID to use.
            num_foreground_points: Maximum foreground points per category.
            num_background_points: Background points per category.
            num_grid_cells: Number of grid cells for prompt generation.
            point_selection_threshold: Minimum feature similarity for a pixel to be
                selected as a foreground point prompt for SAM. Used during prompt
                generation to identify candidate object locations. Higher values =
                fewer, more confident point proposals.
            confidence_threshold: Minimum confidence score for keeping predicted masks
                in the final output. Computed as a weighted combination of SAM's IoU
                prediction and mean similarity within the mask region. Higher values =
                stricter filtering, fewer masks.
            precision: Model precision ("bf16", "fp32").
            compile_models: Whether to compile models with torch.compile.
            device: Device for inference.
            postprocessor: Post-processor applied after predict().
                Defaults to :func:`~instantlearn.components.postprocessing.default_postprocessor`
                (MaskIoMNMS + BoxIoMNMS).
        """
        if postprocessor is None:
            postprocessor = default_postprocessor()
        super().__init__(device=device, precision=precision, postprocessor=postprocessor)
        self.sam_predictor = load_sam_model(
            sam,
            device=device,
            precision=precision,
            compile_models=compile_models,
        )

        self.encoder = ImageEncoder(
            model_id=encoder_model,
            backend=Backend.HUGGINGFACE,
            device=device,
            precision=precision,
            compile_models=compile_models,
        )

        self.masked_feature_extractor = MaskedFeatureExtractor(
            input_size=self.encoder.input_size,
            patch_size=self.encoder.patch_size,
            device=device,
        )

        self.similarity_matcher = CosineSimilarity(feature_size=self.encoder.feature_size)

        max_points = num_foreground_points + num_background_points
        self.prompt_generator = GridPromptGenerator(
            num_grid_cells=num_grid_cells,
            point_selection_threshold=point_selection_threshold,
            num_bg_points=num_background_points,
            num_foreground_points=num_foreground_points,
            max_points=max_points,
        )

        self.segmenter = SamDecoder(
            sam_predictor=self.sam_predictor,
            confidence_threshold=confidence_threshold,
        )

        # Reference features (set during fit).
        self.ref_features: ReferenceFeatures | None = None
        # Category id -> label name mapping (set during fit), used to build
        # ``Prediction.label_names`` at the numpy boundary.
        self._category_names: dict[int, str] = {}

    @classmethod
    def card(cls) -> ModelCard:
        """Return the static capability descriptor for PerDino."""
        return _PERDINO_CARD

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
        reference_embeddings = self.encoder(reference_batch.images)
        self.ref_features = self.masked_feature_extractor(
            reference_embeddings,
            reference_batch.masks,
            reference_batch.label_ids,
        )
        # Cache category id -> name so predict() can build Prediction.label_names.
        self._category_names = {}
        for sample in reference_batch.samples:
            if not sample.label_ids or not sample.category_labels:
                continue
            for cat_id, label in zip(sample.label_ids, sample.category_labels, strict=False):
                self._category_names.setdefault(int(cat_id), label)

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
            msg = "PerDino requires fit() before predict(). Call model.fit(reference_sample) first."
            raise ModelNotFittedError(msg)

        # Convert inputs at the single torch boundary. ``Sample.image`` is numpy
        # HWC, so sizes must come from the tensor shape — calling ``.size()`` on
        # the raw numpy array would raise.
        tensor_batch = batch_to_tensors(target_batch, device=str(self.ref_features.device))
        original_sizes = torch.tensor(
            [image.shape[-2:] for image in tensor_batch.images],
            device=self.ref_features.device,
        )

        # Encode targets [T, num_patches, embed_dim]
        target_embeddings = self.encoder(target_batch.images)

        # Compute similarities [T, C, feat_size, feat_size]
        similarities = self.similarity_matcher(
            self.ref_features.masked_ref_embeddings,
            target_embeddings,
            self.ref_features.category_ids,
        )

        # Generate prompts [T, C, max_points, 4]
        point_prompts = self.prompt_generator(
            similarities,
            self.ref_features.category_ids,
            original_sizes,
        )

        # Decode masks for all images. ``SamDecoder`` normalizes the images to CHW tensors on the SAM
        # device/dtype internally, so pass the converted ``tensor_batch`` (CHW).
        predictions = self.segmenter(
            tensor_batch.images,
            self.ref_features.category_ids,
            point_prompts=point_prompts,
            similarities=similarities,
        )
        predictions = apply_postprocessing(predictions, self.postprocessor)
        return [dict_to_prediction(pred, self._category_names) for pred in predictions]

    @torch.no_grad()
    def _build_inference_graph(
        self,
        export_device: torch.device,
        *,
        sam_hq_tiny_fallback: bool,
    ) -> PerDinoInferenceGraph:
        """Build the traceable inference graph with baked reference features.

        Args:
            export_device: Device to place the graph and reference tensors on.
            sam_hq_tiny_fallback: When ``True`` and the configured SAM variant is
                SAM-HQ-Tiny (non-deterministic under OpenVINO), fall back to
                SAM-HQ-base for the exported graph.

        Returns:
            An FP32 ``PerDinoInferenceGraph`` ready for tracing.

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
            )

        self.sam_predictor.sync_device(export_device, dtype=torch.float32)
        self.segmenter.device = self.sam_predictor.device
        ref_features = self.ref_features.to(export_device)
        export_decoder = fallback_segmenter if fallback_segmenter is not None else self.segmenter

        # Force FP32 for stable CPU tracing.
        return (
            PerDinoInferenceGraph(
                encoder=EncoderForwardFeaturesWrapper(
                    self.encoder._model.model,  # noqa: SLF001
                    ignore_token_length=self.encoder._model.ignore_token_length,  # noqa: SLF001
                    input_size=self.encoder.input_size,
                ),
                similarity_matcher=self.similarity_matcher,
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
        """Export the baked PerDino graph to ONNX.

        Requires ``fit()`` first — the reference features are baked into the
        exported graph as constants.

        Args:
            export_path: Destination directory. ``None`` writes to a temporary
                directory that is *not* auto-deleted.
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
        """Export the baked PerDino graph to an OpenVINO IR directory.

        Conversion goes Torch -> ONNX -> OpenVINO IR. Requires ``fit()`` first —
        reference features are baked into the graph. Also writes a
        ``metadata.json`` so :class:`PerDinoOpenVINO` can build
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
                "INT4 compressed models for PerDino produce random noisy masks and are not accurate. "
                "Please use INT8 compression or no compression for PerDino exports."
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

        write_metadata(export_dir, self.encoder.input_size, self.encoder.patch_size, self._category_names)
        return export_dir
