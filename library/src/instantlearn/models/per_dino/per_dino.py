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
from instantlearn.components.sam import load_sam_model
from instantlearn.data.base.batch import Batch, Collatable
from instantlearn.data.base.sample import Sample
from instantlearn.models.base import Model
from instantlearn.models.matcher.matcher import EncoderForwardFeaturesWrapper
from instantlearn.utils.constants import Backend, CompressionMode, SAMModelName
from instantlearn.utils.graph_export import export_inference_graph

from .prompt_generators import GridPromptGenerator

logger = logging.getLogger(__name__)


class PerDinoInferenceGraph(nn.Module):
    """Traceable inference graph with frozen reference features for ONNX export.

    Mirrors :meth:`PerDino.predict` for a single image, but keeps every step
    traceable so the whole pipeline (encoder → similarity → prompts → SAM →
    post-processing) becomes one self-contained ONNX/OpenVINO graph.
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

        # Register the post-processor as a submodule so its parameters are
        # captured during tracing.
        self.add_module("export_postprocessor", postprocessor)

        self.register_buffer("masked_ref_embeddings", ref_features.masked_ref_embeddings)
        self.register_buffer("category_ids", torch.tensor(ref_features.category_ids, device=ref_features.device))

    def forward(self, target_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single image forward pass: target_image [1, 3, H, W] -> (masks, scores, labels)."""
        target_embeddings = self.encoder(target_image)
        feature_device = target_embeddings.device

        # Align frozen reference tensors to the target device so tracing never
        # produces a mixed-device matmul.
        masked_ref_embeddings = self.masked_ref_embeddings.to(feature_device)
        category_ids = self.category_ids.to(feature_device)

        # scalar_tensor keeps the original size dynamic instead of baking in the
        # trace-time value.
        height = torch.scalar_tensor(target_image.shape[2], dtype=torch.long, device=feature_device)
        width = torch.scalar_tensor(target_image.shape[3], dtype=torch.long, device=feature_device)
        original_sizes = torch.stack([height, width], dim=0).unsqueeze(0)

        similarities = self.similarity_matcher(masked_ref_embeddings, target_embeddings, category_ids)
        point_prompts = self.prompt_generator(similarities, category_ids, original_sizes)

        masks, scores, labels = self.sam_decoder.forward_export(
            target_image[0],
            category_ids,
            point_prompts[0],
            similarities[0],
        )

        if self.export_postprocessor is not None:
            masks, scores, labels = self.export_postprocessor(masks, scores, labels)

        return masks, scores, labels


class PerDino(Model):
    """PerDino algorithm model for one-shot segmentation.

    Matches reference objects to target images by comparing features extracted by DINOv2
    using cosine similarity. A grid prompt generator creates multi-object aware prompts.

    The pipeline is fully traceable (ONNX/TorchScript compatible):
    - Encoder → MaskedFeatureExtractor → CosineSimilarity → GridPromptGenerator → SamDecoder

    Examples:
        >>> from instantlearn.models import PerDino
        >>> from instantlearn.data.base import Batch
        >>> from instantlearn.data.base.sample import Sample
        >>> import torch
        >>> import numpy as np

        >>> perdino = PerDino()

        >>> # Create reference sample
        >>> ref_sample = Sample(
        ...     image=torch.zeros((3, 1024, 1024)),
        ...     masks=torch.ones(30, 30, dtype=torch.bool).unsqueeze(0),
        ...     category_ids=np.array([1]),
        ...     is_reference=[True],
        ...     categories=["object"],
        ... )
        >>> ref_batch = Batch.collate([ref_sample])

        >>> # Create target sample
        >>> target_sample = Sample(
        ...     image=torch.zeros((3, 1024, 1024)),
        ...     is_reference=[False],
        ...     categories=["object"],
        ... )
        >>> target_batch = Batch.collate([target_sample])

        >>> # Run fit and predict
        >>> perdino.fit(ref_batch)
        >>> predict_results = perdino.predict(target_batch)
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
        num_export_instances: int = 8,
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
            num_export_instances: Maximum instances per category the **exported**
                (ONNX/OpenVINO) model can detect. Each slot costs one SAM decoder pass.
                Only affects export; the PyTorch path is unbounded. Default: 8.
        """
        if postprocessor is None:
            postprocessor = default_postprocessor()
        super().__init__(postprocessor=postprocessor)
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
            num_export_instances=num_export_instances,
            num_background_points=num_background_points,
        )

        self.ref_features: ReferenceFeatures | None = None

    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Learn from reference images.

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
            reference_batch.category_ids,
        )

    def predict(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Predict masks for target images.

        Args:
            target: Target data to infer. Accepts:
                - Sample: A single target sample
                - list[Sample]: A list of target samples
                - Batch: A batch of target samples
                - str | Path: A single image path
                - list[str] | list[Path]: Multiple image paths

        Returns:
            List of predictions per image, each containing:
                "pred_masks": [num_masks, H, W]
                "pred_scores": [num_masks]
                "pred_labels": [num_masks] - category IDs

        Raises:
            RuntimeError: If reference features are not available.
        """
        target_batch = Batch.collate(target)
        if self.ref_features is None:
            msg = "No reference features. Call fit() first."
            raise RuntimeError(msg)

        # Get original sizes [T, 2]
        original_sizes = torch.tensor(
            [image.size()[-2:] for image in target_batch.images],
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

        # Generate prompts [T, C, max_points, 4], [T, C]
        point_prompts = self.prompt_generator(
            similarities,
            self.ref_features.category_ids,
            original_sizes,
        )

        # Decode masks
        predictions = self.segmenter(
            target_batch.images,
            self.ref_features.category_ids,
            point_prompts=point_prompts,
            similarities=similarities,
        )
        return self.apply_postprocessing(predictions)

    @torch.no_grad()
    def export(
        self,
        export_dir: str | Path = Path("./exports/per_dino"),
        backend: str | Backend = Backend.ONNX,
        compression: CompressionMode = CompressionMode.INT8_SYM,
    ) -> Path:
        """Export the model to ONNX or OpenVINO.

        The exported graph detects at most ``num_export_instances`` objects per
        category (see :meth:`__init__`), whereas the PyTorch path is unbounded.

        Args:
            export_dir: Directory to save exported models.
            backend: Export backend (ONNX, OpenVINO).
            compression: Weight compression mode for the exported OpenVINO model.
                See :class:`~instantlearn.utils.constants.CompressionMode` for options.
                Only applied when *backend* is ``OPENVINO``. Default: INT8_SYM.

        Returns:
            Path to the exported model file.

        Raises:
            RuntimeError: If fit() has not been called first.
        """
        if self.ref_features is None:
            msg = "No reference features. Call fit() first."
            raise RuntimeError(msg)

        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        # SAM-HQ-Tiny has non-deterministic layers that make the exported graph
        # unreliable, so OpenVINO exports fall back to SAM-HQ-base.
        export_decoder = self.segmenter
        if Backend(backend) == Backend.OPENVINO and self.sam_predictor.sam_model_name == SAMModelName.SAM_HQ_TINY:
            logger.warning(
                "SAM-HQ-Tiny is not supported for OpenVINO export. "
                "Some of the layers are non-deterministic and so the exported model is not reliable for inference. "
                "Falling back to SAM-HQ-base for the exported model. "
                "SAM-HQ-base weights will be downloaded if not already cached.",
            )
            export_decoder = SamDecoder(
                sam_predictor=load_sam_model(SAMModelName.SAM_HQ_BASE, device="cpu", precision="fp32"),
                confidence_threshold=self.segmenter.confidence_threshold,
                num_export_instances=self.segmenter.num_export_instances,
                num_background_points=self.segmenter.num_background_points,
            )

        export_device = torch.device("cpu") if Backend(backend) == Backend.OPENVINO else self.ref_features.device
        self.sam_predictor.sync_device(export_device, dtype=torch.float32)
        self.segmenter.device = self.sam_predictor.device

        graph = (
            PerDinoInferenceGraph(
                encoder=EncoderForwardFeaturesWrapper(
                    self.encoder._model.model,  # noqa: SLF001
                    ignore_token_length=self.encoder._model.ignore_token_length,  # noqa: SLF001
                ),
                similarity_matcher=self.similarity_matcher,
                prompt_generator=self.prompt_generator,
                sam_decoder=export_decoder,
                ref_features=self.ref_features.to(export_device),
                postprocessor=self.postprocessor,
            )
            .to(export_device)
            .float()
        )  # Force FP32 for stable CPU tracing

        return export_inference_graph(
            graph=graph,
            export_dir=export_path,
            model_name="per_dino",
            input_size=self.encoder.input_size,
            backend=backend,
            compression=compression,
            device=export_device,
        )
