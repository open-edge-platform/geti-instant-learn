# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM3 model for text and visual prompting."""

import logging
from enum import Enum
from itertools import zip_longest

import numpy as np
import torch
from transformers import CLIPTokenizerFast

from instantlearn.data.base.batch import Batch, Collatable
from instantlearn.data.base.sample import Sample
from instantlearn.models.base import Model
from instantlearn.utils import precision_to_torch_dtype

from .model import Sam3Model
from .processing import Sam3Postprocessor, Sam3Preprocessor, Sam3PromptPreprocessor

logger = logging.getLogger(__name__)


class Sam3PromptMode(str, Enum):
    """Prompt mode for SAM3 inference.

    Attributes:
        CLASSIC: Original SAM3 behavior. Text/box prompts are provided per target
            image. Boxes are encoded against the target image's own features.
        VISUAL_EXEMPLAR: Cross-image visual query detection. Box prompts on a
            reference image are encoded during fit() and reused for all target
            images. Enables "draw box on image A → detect similar on images B, C, D".
    """

    CLASSIC = "classic"
    VISUAL_EXEMPLAR = "visual_exemplar"


class SAM3(Model):
    """SAM3 model for text and visual prompting.

    This model uses SAM3 (Segment Anything Model 3) for zero-shot segmentation
    using either text prompts or visual prompts (bounding boxes).

    **Important: SAM3 differs from other prompt-based models** in that it does NOT
    require a separate learning phase. Instead, it performs zero-shot segmentation
    directly during inference using:
    - Text prompts (category names) provided via ``fit()`` or per-sample ``categories``, OR
    - Visual prompts (bounding boxes) provided in the ``bboxes`` field of each sample

    At least one of these prompt types must be provided for each sample during inference.

    NOTE: Currently, SAM3 does not work well with torch.bfloat16 precision.

    Prompt Modes:
        **CLASSIC** (default): Original SAM3 behavior. Text/box prompts are
        provided per target image. ``fit()`` only stores category names.

        **VISUAL_EXEMPLAR**: Cross-image visual query detection. During ``fit()``,
        box/point prompts on reference images are encoded into geometry features and
        cached. During ``predict()``, these cached features are reused for each
        target image — no boxes/points needed on targets.

    Examples:
        >>> from instantlearn.models import SAM3
        >>> from instantlearn.models.sam3.sam3 import Sam3PromptMode
        >>> from instantlearn.data.base import Batch
        >>> from instantlearn.data.base.sample import Sample
        >>> import torch
        >>> import numpy as np

        >>> # Classic mode (default)
        >>> sam3 = SAM3()
        >>> ref_sample = Sample(categories=["shoe", "person"], category_ids=[0, 1])
        >>> sam3.fit(ref_sample)
        >>> results = sam3.predict(Sample(image=torch.zeros((3, 1024, 1024))))

        >>> # Visual exemplar mode with boxes
        >>> sam3_ve = SAM3(prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR)
        >>> ref_sample = Sample(
        ...     image=torch.zeros((3, 1024, 1024)),
        ...     bboxes=np.array([[100, 100, 200, 200]]),  # [x1, y1, x2, y2] on reference
        ...     category_ids=np.array([0]),
        ... )
        >>> sam3_ve.fit(ref_sample)
        >>> results = sam3_ve.predict(Sample(image=torch.zeros((3, 1024, 1024))))

        >>> # Visual exemplar mode with points
        >>> sam3_pt = SAM3(prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR)
        >>> ref_sample = Sample(
        ...     image=torch.zeros((3, 1024, 1024)),
        ...     points=np.array([[150, 150]]),  # [x, y] on reference
        ...     category_ids=np.array([0]),
        ... )
        >>> sam3_pt.fit(ref_sample)
        >>> results = sam3_pt.predict(Sample(image=torch.zeros((3, 1024, 1024))))
    """

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        resolution: int = 1008,
        precision: str = "fp32",
        compile_models: bool = False,
        prompt_mode: Sam3PromptMode | str = Sam3PromptMode.CLASSIC,
        drop_spatial_bias: bool = True,
    ) -> None:
        """Initialize the SAM3 model.

        Args:
            device: The device to use ('cuda', 'xpu', or 'cpu').
            confidence_threshold: The confidence threshold for filtering predictions.
            resolution: The input image resolution.
            precision: The precision to use for the model ('bf16' or 'fp32').
            compile_models: Whether to compile the models.
            prompt_mode: Prompt mode for inference. 'classic' for original SAM3
                behavior, 'visual_exemplar' for cross-image visual query detection.
            drop_spatial_bias: When True and in VISUAL_EXEMPLAR mode, skip
                coordinate projection and position encoding in the geometry
                encoder, keeping only ROI-pooled visual features. This removes
                spatial bias from the reference image position. Default: True.
        """
        super().__init__()

        self.device = device
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution
        self.precision = precision
        self.compile_models = compile_models
        self.prompt_mode = Sam3PromptMode(prompt_mode)
        self.drop_spatial_bias = drop_spatial_bias

        # Category mapping from fit() - optional for consistency with GroundedSAM
        self.category_mapping: dict[str, int] | None = None

        # Visual exemplar cached features (set during fit in VISUAL_EXEMPLAR mode)
        self.exemplar_geometry_features: list[torch.Tensor] | None = None
        self.exemplar_geometry_mask: list[torch.Tensor] | None = None
        self.exemplar_text_features: list[torch.Tensor] | None = None
        self.exemplar_text_mask: list[torch.Tensor] | None = None
        self.exemplar_category_ids: list[int] | None = None

        # Preprocessors and postprocessor
        self.image_preprocessor = Sam3Preprocessor(target_size=resolution).to(device)
        self.prompt_preprocessor = Sam3PromptPreprocessor(target_size=resolution).to(device)
        self.postprocessor = Sam3Postprocessor(
            target_size=resolution,
            threshold=confidence_threshold,
            mask_threshold=0.5,
        ).to(device)

        # Tokenizer for text prompts (still from transformers, but not used in ONNX path)
        self.tokenizer = CLIPTokenizerFast.from_pretrained("facebook/sam3")

        self.model = (
            Sam3Model.from_pretrained(
                "facebook/sam3",
                torch_dtype=precision_to_torch_dtype(precision),
            )
            .to(device)
            .eval()
        )

    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Learn from reference samples.

        In CLASSIC mode, stores category mapping only (no image processing).
        In VISUAL_EXEMPLAR mode, encodes box/point prompts on reference images into
        geometry features and caches them for reuse during predict().

        Args:
            reference: Reference data to learn from. Accepts:
                - Sample: A single reference sample
                - list[Sample]: A list of reference samples
                - Batch: A batch of reference samples

        Raises:
            ValueError: If in VISUAL_EXEMPLAR mode and no bboxes or points are provided
                in any reference sample.
        """
        reference_batch = Batch.collate(reference)

        if self.prompt_mode == Sam3PromptMode.CLASSIC:
            self._fit_classic(reference_batch)
        else:
            self._fit_visual_exemplar(reference_batch)

    def _fit_classic(self, reference_batch: Batch) -> None:
        """Store category mapping from reference batch.

        Args:
            reference_batch: Batch of reference samples.
        """
        self.category_mapping = {}
        for sample in reference_batch.samples:
            if sample.categories is None or sample.category_ids is None:
                continue
            for category_id, category in zip(sample.category_ids, sample.categories, strict=False):
                if category not in self.category_mapping:
                    self.category_mapping[category] = int(category_id)

    @torch.no_grad()
    def _fit_visual_exemplar(self, reference_batch: Batch) -> None:
        """Encode visual exemplar features from reference images and boxes/points.

        For each reference sample with bounding boxes or points, encodes the regions
        using the GeometryEncoder against the reference image's ViT features.
        Results are cached for reuse in predict().

        Args:
            reference_batch: Batch of reference samples with images and bboxes/points.

        Raises:
            ValueError: If no reference samples contain bboxes or points.
        """
        all_geometry_features: list[torch.Tensor] = []
        all_geometry_masks: list[torch.Tensor] = []
        all_category_ids: list[int] = []
        all_text_prompts: list[str] = []

        for sample in reference_batch.samples:
            bboxes = sample.bboxes
            points = sample.points
            has_bboxes = bboxes is not None and not (isinstance(bboxes, np.ndarray) and bboxes.size == 0)
            has_points = points is not None and not (isinstance(points, np.ndarray) and points.size == 0)

            if not has_bboxes and not has_points:
                continue
            if sample.image is None:
                msg = "VISUAL_EXEMPLAR mode requires images in reference samples."
                raise ValueError(msg)

            # Preprocess reference image
            image_tensor = sample.image.unsqueeze(0) if sample.image.ndim == 3 else sample.image
            pixel_values, original_sizes = self.image_preprocessor(image_tensor.to(self.device))
            vision_embeds = self.model.get_vision_features(pixel_values)

            fpn_hidden_states = vision_embeds["fpn_hidden_states"][:-1]
            fpn_position_encoding = vision_embeds["fpn_position_encoding"][:-1]

            # Determine number of prompts and build aligned lists
            num_prompts = max(len(bboxes) if has_bboxes else 0, len(points) if has_points else 0)
            categories = sample.categories if sample.categories is not None else ["visual"] * num_prompts
            category_ids = sample.category_ids if sample.category_ids is not None else [0] * num_prompts

            # Encode each box with its center point to get per-exemplar features
            # The geometry encoder concatenates box ROI features + center point features
            if has_bboxes:
                for bbox, category, cat_id in zip(bboxes, categories, category_ids, strict=True):
                    input_boxes, _ = self.prompt_preprocessor(original_sizes, input_boxes=bbox)
                    input_boxes_labels = torch.ones((1, 1), dtype=torch.long, device=self.device)

                    box_embeddings = input_boxes.to(dtype=fpn_hidden_states[0].dtype)
                    box_mask = torch.ones(1, 1, dtype=torch.bool, device=self.device)

                    # Derive center point from box: [(x1+x2)/2, (y1+y2)/2]
                    center_point = (input_boxes[..., :2] + input_boxes[..., 2:]) / 2  # (1, 1, 2)
                    point_embeddings = center_point.to(dtype=fpn_hidden_states[0].dtype)
                    point_mask = torch.ones(1, 1, dtype=torch.bool, device=self.device)
                    point_labels = torch.ones((1, 1), dtype=torch.long, device=self.device)

                    geometry_outputs = self.model.geometry_encoder(
                        box_embeddings=box_embeddings,
                        box_mask=box_mask,
                        box_labels=input_boxes_labels,
                        point_embeddings=point_embeddings,
                        point_mask=point_mask,
                        point_labels=point_labels,
                        img_feats=fpn_hidden_states,
                        img_pos_embeds=fpn_position_encoding,
                        drop_spatial_bias=self.drop_spatial_bias,
                    )

                    all_geometry_features.append(geometry_outputs["last_hidden_state"])
                    all_geometry_masks.append(geometry_outputs["attention_mask"])
                    all_category_ids.append(int(cat_id))
                    all_text_prompts.append(category)

            # Encode standalone points (when no box is provided)
            if has_points and not has_bboxes:
                for point, category, cat_id in zip(points, categories, category_ids, strict=True):
                    _, input_points = self.prompt_preprocessor(original_sizes, input_points=point)
                    input_points_labels = torch.ones((1, 1), dtype=torch.long, device=self.device)

                    point_embeddings = input_points.to(dtype=fpn_hidden_states[0].dtype)
                    point_mask = torch.ones(1, 1, dtype=torch.bool, device=self.device)

                    geometry_outputs = self.model.geometry_encoder(
                        point_embeddings=point_embeddings,
                        point_mask=point_mask,
                        point_labels=input_points_labels,
                        img_feats=fpn_hidden_states,
                        img_pos_embeds=fpn_position_encoding,
                        drop_spatial_bias=self.drop_spatial_bias,
                    )

                    all_geometry_features.append(geometry_outputs["last_hidden_state"])
                    all_geometry_masks.append(geometry_outputs["attention_mask"])
                    all_category_ids.append(int(cat_id))
                    all_text_prompts.append(category)

        if not all_geometry_features:
            msg = "VISUAL_EXEMPLAR mode requires at least one reference sample with bboxes or points."
            raise ValueError(msg)

        # Cache geometry features (each exemplar is [1, num_prompts, 256])
        self.exemplar_geometry_features = all_geometry_features
        self.exemplar_geometry_mask = all_geometry_masks
        self.exemplar_category_ids = all_category_ids

        # Pre-compute text features per unique text prompt, then map per exemplar
        unique_prompts = list(dict.fromkeys(all_text_prompts))  # preserve order, deduplicate
        text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for prompt in unique_prompts:
            text_inputs = self.tokenizer([prompt], return_tensors="pt", padding="max_length", max_length=32)
            input_ids = text_inputs.input_ids.to(self.device)
            attention_mask = text_inputs.attention_mask.to(self.device)
            text_outputs = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_cache[prompt] = (text_outputs.pooler_output, attention_mask.bool())

        # Store per-exemplar text features aligned with geometry features
        self.exemplar_text_features = [text_cache[p][0] for p in all_text_prompts]
        self.exemplar_text_mask = [text_cache[p][1] for p in all_text_prompts]

        # Also store category mapping if categories are available
        self.category_mapping = {}
        for sample in reference_batch.samples:
            if sample.categories is None or sample.category_ids is None:
                continue
            for category_id, category in zip(sample.category_ids, sample.categories, strict=False):
                if category not in self.category_mapping:
                    self.category_mapping[category] = int(category_id)

        logger.info(
            "Cached %d visual exemplar(s): prompts=%s, category_ids=%s",
            len(all_category_ids),
            all_text_prompts,
            all_category_ids,
        )

    @staticmethod
    def _aggregate_results(
        all_masks: list[torch.Tensor],
        all_boxes: list[torch.Tensor],
        all_labels: list[torch.Tensor],
        img_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """Aggregate results from multiple predictions.

        Args:
            all_masks: List of mask tensors.
            all_boxes: List of box tensors.
            all_labels: List of labels.
            img_size: The image size (height, width).

        Returns:
            Dictionary with aggregated predictions.
        """
        # Filter out empty tensors before concatenation
        non_empty_masks = [masks for masks in all_masks if masks.numel() > 0]
        non_empty_boxes = [boxes for boxes in all_boxes if boxes.numel() > 0]
        non_empty_labels = [labels for labels in all_labels if labels.numel() > 0]

        if non_empty_masks:
            aggregated_masks = torch.cat(non_empty_masks, dim=0)
            aggregated_boxes = torch.cat(non_empty_boxes, dim=0)
            aggregated_labels = torch.cat(non_empty_labels, dim=0)
        else:
            # No predictions found
            aggregated_masks = torch.empty(0, *img_size)
            aggregated_boxes = torch.empty(0, 5)
            aggregated_labels = torch.empty(0, dtype=torch.long)

        return {
            "pred_masks": aggregated_masks,
            "pred_boxes": aggregated_boxes,
            "pred_labels": aggregated_labels,
        }

    def predict(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Perform inference on target images.

        In CLASSIC mode, processes text/box prompts per target image.
        In VISUAL_EXEMPLAR mode, reuses cached exemplar features from fit().

        Args:
            target: Target data to infer. Accepts:
                - Sample: A single target sample
                - list[Sample]: A list of target samples
                - Batch: A batch of target samples
                - str | Path: A single image path
                - list[str] | list[Path]: Multiple image paths

        Returns:
            List of prediction dicts per image with 'pred_masks', 'pred_boxes',
            'pred_labels'.

        Raises:
            RuntimeError: If in VISUAL_EXEMPLAR mode and fit() has not been called.
        """
        if self.prompt_mode == Sam3PromptMode.VISUAL_EXEMPLAR:
            return self._predict_visual_exemplar(target)
        return self._predict_classic(target)

    def _predict_classic(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Classic SAM3 prediction with per-image text/box/point prompts.

        Args:
            target: Target data to infer.

        Returns:
            List of prediction dicts per image.
        """
        target_batch = Batch.collate(target)
        results = []
        samples = target_batch.samples

        # Use stored categories from fit() if available, otherwise use per-sample
        use_fitted_categories = self.category_mapping is not None

        # Process each image's prompts individually
        for sample in samples:
            img_size = sample.image.shape[-2:]
            bboxes = sample.bboxes if sample.bboxes is not None else []
            points = sample.points if sample.points is not None else []

            # Preprocess image
            image_tensor = sample.image.unsqueeze(0) if sample.image.ndim == 3 else sample.image
            with torch.no_grad():
                pixel_values, original_sizes = self.image_preprocessor(image_tensor.to(self.device))
                vision_embeds = self.model.get_vision_features(pixel_values)

            # Determine text prompts and category IDs
            if use_fitted_categories:
                texts = list(self.category_mapping.keys())
                category_ids = list(self.category_mapping.values())
            else:
                texts = sample.categories or []
                category_ids = sample.category_ids
                # Use "visual" placeholder when only bboxes/points are provided
                num_visual_prompts = max(len(bboxes), len(points))
                if num_visual_prompts and len(texts) != num_visual_prompts:
                    texts = ["visual"] * num_visual_prompts

            all_masks: list[torch.Tensor] = []
            all_boxes: list[torch.Tensor] = []
            all_labels: list[torch.Tensor] = []

            for text, bbox, point, cat_id in zip_longest(texts, bboxes, points, category_ids, fillvalue=None):
                # Tokenize text prompt (default to "visual" for visual-only prompts)
                text_inputs = self.tokenizer([text or "visual"], return_tensors="pt", padding="max_length", max_length=32)
                input_ids = text_inputs.input_ids.to(self.device)
                attention_mask = text_inputs.attention_mask.to(self.device)

                # Prepare box inputs if bbox is provided (xyxy format)
                input_boxes = None
                input_boxes_labels = None
                if bbox is not None:
                    input_boxes, _ = self.prompt_preprocessor(original_sizes, input_boxes=bbox)
                    input_boxes_labels = torch.ones((1, 1), dtype=torch.long, device=self.device)

                # Prepare point inputs if point is provided (xy format)
                input_points = None
                input_points_labels = None
                if point is not None:
                    _, input_points = self.prompt_preprocessor(original_sizes, input_points=point)
                    input_points_labels = torch.ones((1, 1), dtype=torch.long, device=self.device)

                with torch.no_grad():
                    outputs = self.model(
                        vision_embeds=vision_embeds,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        input_boxes=input_boxes,
                        input_boxes_labels=input_boxes_labels,
                        input_points=input_points,
                        input_points_labels=input_points_labels,
                    )

                # Postprocess
                result = self.postprocessor(outputs, target_sizes=[img_size])
                boxes_with_scores = torch.cat(
                    [result[0]["boxes"], result[0]["scores"].unsqueeze(1)],
                    dim=1,
                )
                all_masks.append(result[0]["masks"].cpu())
                all_boxes.append(boxes_with_scores.cpu())
                all_labels.append(torch.full((len(result[0]["boxes"]),), cat_id, dtype=torch.int64))

            results.append(self._aggregate_results(all_masks, all_boxes, all_labels, img_size))

        return results

    def _predict_visual_exemplar(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Visual exemplar prediction using cached geometry features from fit().

        For each target image, reuses the cached exemplar geometry features
        (extracted from reference images during fit) as prompt conditioning.

        Args:
            target: Target data to infer.

        Returns:
            List of prediction dicts per image.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self.exemplar_geometry_features is None:
            msg = "No cached exemplar features. Call fit() with reference images and bboxes first."
            raise RuntimeError(msg)

        target_batch = Batch.collate(target)
        results = []

        for sample in target_batch.samples:
            img_size = sample.image.shape[-2:]

            # Preprocess target image
            image_tensor = sample.image.unsqueeze(0) if sample.image.ndim == 3 else sample.image
            with torch.no_grad():
                pixel_values, original_sizes = self.image_preprocessor(image_tensor.to(self.device))
                vision_embeds = self.model.get_vision_features(pixel_values)

            all_masks: list[torch.Tensor] = []
            all_boxes: list[torch.Tensor] = []
            all_labels: list[torch.Tensor] = []

            # Run detection for each cached exemplar
            for i, (geo_feats, geo_mask, text_feats, text_mask, cat_id) in enumerate(
                zip(
                    self.exemplar_geometry_features,
                    self.exemplar_geometry_mask,
                    self.exemplar_text_features,
                    self.exemplar_text_mask,
                    self.exemplar_category_ids,
                    strict=True,
                )
            ):
                with torch.no_grad():
                    outputs = self.model(
                        vision_embeds=vision_embeds,
                        text_embeds=text_feats,
                        attention_mask=text_mask.long(),
                        precomputed_geometry_features=geo_feats,
                        precomputed_geometry_mask=geo_mask,
                    )

                # Postprocess
                result = self.postprocessor(outputs, target_sizes=[img_size])
                boxes_with_scores = torch.cat(
                    [result[0]["boxes"], result[0]["scores"].unsqueeze(1)],
                    dim=1,
                )
                all_masks.append(result[0]["masks"].cpu())
                all_boxes.append(boxes_with_scores.cpu())
                all_labels.append(torch.full((len(result[0]["boxes"]),), cat_id, dtype=torch.int64))

            results.append(self._aggregate_results(all_masks, all_boxes, all_labels, img_size))

        return results
