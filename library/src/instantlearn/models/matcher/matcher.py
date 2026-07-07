# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Matcher model, based on the paper 'Segment Anything with One Shot Using All-Purpose Feature Matching'."""

import logging

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
from instantlearn.models.model_card import ModelCard
from instantlearn.models.torch_adapter import tensors_to_prediction
from instantlearn.models.torch_base import TorchModel
from instantlearn.utils.constants import Backend, PromptType, SAMModelName, ShotMode
from instantlearn.utils.errors import ModelNotFittedError

from .prompt_generators import BidirectionalPromptGenerator

logger = logging.getLogger(__name__)


class EncoderForwardFeaturesWrapper(nn.Module):
    """Wrapper for image encoder to expose forward_features method for export."""

    def __init__(
        self,
        encoder: nn.Module,
        ignore_token_length: int,
        input_size: int = 512,
    ) -> None:
        """Initialize the encoder wrapper.

        Args:
            encoder: The underlying encoder module.
            ignore_token_length: Number of tokens to ignore.
            input_size: Input image size.
        """
        super().__init__()
        self.encoder = encoder
        self.ignore_token_length = ignore_token_length
        self.input_size = input_size
        self.register_buffer("IMAGENET_DEFAULT_MEAN", torch.tensor((0.485, 0.456, 0.406)))
        self.register_buffer("IMAGENET_DEFAULT_STD", torch.tensor((0.229, 0.224, 0.225)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get encoder features."""
        imagenet_mean = self.IMAGENET_DEFAULT_MEAN.to(device=x.device, dtype=x.dtype)
        imagenet_std = self.IMAGENET_DEFAULT_STD.to(device=x.device, dtype=x.dtype)
        x = x.float() / 255.0
        x = functional.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear")
        x = (x - imagenet_mean[None, :, None, None]) / imagenet_std[None, :, None, None]
        features = self.encoder.forward_features(x)
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

        # Get original size from input tensor [1, 3, H, W] using public APIs only.
        # scalar_tensor preserves dynamic shape in export without relying on private/legacy ONNX helpers.
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
        >>> import torch
        >>> import numpy as np

        >>> matcher = Matcher()

        >>> # Create reference sample
        >>> ref_sample = Sample(
        ...     image=torch.zeros((3, 1024, 1024)),
        ...     masks=torch.ones(30, 30, dtype=torch.bool).unsqueeze(0),
        ...     is_reference=[True],
        ...     categories=[Category(1, "object")],
        ... )
        >>> ref_batch = Batch.collate([ref_sample])

        >>> # Create target sample
        >>> target_sample = Sample(
        ...     image=torch.zeros((3, 1024, 1024)),
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
        device: str = "cuda",
        postprocessor: PostProcessor | None = None,
        similarity_threshold: float | None = None,
        num_grid_cells: int = 8,
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
            device: Device for inference.
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
        """
        if postprocessor is None:
            postprocessor = default_postprocessor()
        super().__init__(device=device, precision=precision, postprocessor=postprocessor)
        # SAM predictor
        self.sam_predictor = load_sam_model(
            sam,
            device=device,
            precision=precision,
            compile_models=compile_models,
        )

        # Image encoder
        self.encoder = ImageEncoder(
            model_id=encoder_model,
            backend=Backend.HUGGINGFACE,
            device=device,
            precision=precision,
            compile_models=compile_models,
        )

        # Feature extractor
        self.masked_feature_extractor = MaskedFeatureExtractor(
            input_size=self.encoder.input_size,
            patch_size=self.encoder.patch_size,
            device=device,
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
        )

        # Reference features (set during fit)
        self.ref_features: ReferenceFeatures | None = None
        # Category id -> label name mapping (set during fit), used to build
        # ``Prediction.label_names`` at the numpy boundary.
        self._category_names: dict[int, str] = {}

    @classmethod
    def card(cls) -> ModelCard:
        """Return the static capability descriptor for Matcher."""
        return ModelCard(
            name="Matcher",
            family="matcher",
            description="One-shot mask matcher (DINOv3 features + SAM decoder)",
            prompt_types=frozenset({PromptType.MASK}),
            shot_modes=frozenset({ShotMode.ONE_SHOT, ShotMode.FEW_SHOT}),
            exportable_to=frozenset({Backend.OPENVINO, Backend.ONNX}),
        )

    @property
    def input_size(self) -> int:
        """Square input size expected by the encoder (e.g. 512)."""
        return self.encoder.input_size

    def fit(self, reference: Sample | list[Sample] | Batch) -> ReferenceFeatures:
        """Learn from reference images.

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
        # Cache category id -> name so predict() can build Prediction.label_names.
        self._category_names = {}
        for sample in reference_batch.samples:
            if not sample.label_ids or not sample.category_labels:
                continue
            for cat_id, label in zip(sample.label_ids, sample.category_labels, strict=False):
                self._category_names.setdefault(int(cat_id), label)
        return self.ref_features

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

        # Get original sizes [T, 2]
        original_sizes = torch.tensor(
            [image.size()[-2:] for image in target_batch.images],
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

        # Decode masks for all images
        predictions = self.segmenter(
            target_batch.images,
            self.ref_features.category_ids,
            point_prompts=point_prompts,
            similarities=similarities,
        )
        predictions = apply_postprocessing(predictions, self.postprocessor)
        return [self._to_prediction(pred) for pred in predictions]

    def _to_prediction(self, pred: dict[str, torch.Tensor]) -> Prediction:
        """Convert a single torch prediction dict to a numpy ``Prediction``.

        Args:
            pred: Dict with ``pred_masks`` ``[N, H, W]``, ``pred_scores`` ``[N]``,
                ``pred_labels`` ``[N]`` and optionally ``pred_boxes`` ``[N, 5]``
                (xyxy + score).

        Returns:
            A ``Prediction`` with ``label_names`` resolved from the categories
            seen during ``fit()``.
        """
        masks = pred["pred_masks"]
        labels = pred["pred_labels"].to(torch.int32)
        scores = pred.get("pred_scores")
        if scores is None:
            scores = torch.ones(masks.shape[0], device=masks.device)

        boxes = None
        if "pred_boxes" in pred and pred["pred_boxes"].numel() > 0:
            boxes = pred["pred_boxes"][:, :4]

        # Build a categories sequence indexed by label id for label_names lookup.
        max_id = max(self._category_names) if self._category_names else -1
        categories = [self._category_names.get(i, str(i)) for i in range(max_id + 1)]

        return tensors_to_prediction(
            masks=masks,
            scores=scores,
            label_ids=labels,
            categories=categories,
            boxes=boxes,
        )
