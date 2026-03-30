# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM3 model for text and visual prompting."""

import logging
from collections import defaultdict
from enum import Enum
from itertools import zip_longest

import numpy as np
import torch
from transformers import CLIPTokenizerFast

from instantlearn.components.negative_prompts import NegativeMaskToPoints
from instantlearn.components.postprocessing import PostProcessor, default_postprocessor
from instantlearn.data.base.batch import Batch, Collatable
from instantlearn.data.base.sample import BACKGROUND_CATEGORY_ID, Sample
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

    Negative Prompts:
        Background masks (``category_id == BACKGROUND_CATEGORY_ID``) provided
        during ``fit()`` are converted to negative point prompts and pre-encoded
        against the reference image. These geometry features are then concatenated
        with positive prompts during ``predict()`` to suppress false positives.

        **Limitation**: When the reference and target images differ significantly
        in resolution or composition (e.g. 2500×2500 reference vs 680×540 target),
        the transferred negative features may be less effective at suppressing
        marginal detections. This is inherent to SAM3's cross-image transfer.

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

        >>> # N-shot: multiple point prompts for the same category (same image)
        >>> sam3_nshot = SAM3(prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR)
        >>> ref_sample = Sample(
        ...     image=torch.zeros((3, 1024, 1024)),
        ...     points=np.array([[100, 100], [200, 300], [400, 500]]),  # 3 shots
        ...     categories=["shoe", "shoe", "shoe"],
        ...     category_ids=np.array([0, 0, 0]),  # same category
        ... )
        >>> sam3_nshot.fit(ref_sample)  # encodes 3 points together
        >>> results = sam3_nshot.predict(Sample(image=torch.zeros((3, 1024, 1024))))

        >>> # N-shot across multiple reference images
        >>> sam3_cross = SAM3(prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR)
        >>> refs = [
        ...     Sample(
        ...         image=torch.zeros((3, 1024, 1024)),
        ...         points=np.array([[100, 100]]),
        ...         categories=["shoe"],
        ...         category_ids=np.array([0]),
        ...     ),
        ...     Sample(
        ...         image=torch.zeros((3, 1024, 1024)),
        ...         points=np.array([[200, 200]]),
        ...         categories=["shoe"],
        ...         category_ids=np.array([0]),  # same category, different image
        ...     ),
        ... ]
        >>> sam3_cross.fit(refs)  # features concatenated across images
        >>> results = sam3_cross.predict(Sample(image=torch.zeros((3, 1024, 1024))))
    """

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        resolution: int = 1008,
        precision: str = "fp32",
        compile_models: bool = False,
        model_id: str = "facebook/sam3",
        prompt_mode: Sam3PromptMode | str = Sam3PromptMode.CLASSIC,
        drop_spatial_bias: bool = False,
        num_negative_points: int = 5,
        postprocessor: PostProcessor | None = None,
    ) -> None:
        """Initialize the SAM3 model.

        Args:
            device: The device to use ('cuda', 'xpu', or 'cpu').
            confidence_threshold: The confidence threshold for filtering predictions.
            resolution: The input image resolution.
            precision: The precision to use for the model ('bf16' or 'fp32').
            compile_models: Whether to compile the models.
            model_id: HuggingFace model ID or local path to load the SAM3 model
                and tokenizer from. Default: "facebook/sam3".
            prompt_mode: Prompt mode for inference. 'classic' for original SAM3
                behavior, 'visual_exemplar' for cross-image visual query detection.
            drop_spatial_bias: When True and in VISUAL_EXEMPLAR mode, skip
                coordinate projection and position encoding in the geometry
                encoder, keeping only ROI-pooled visual features. This removes
                spatial bias from the reference image position. Default: False.
            num_negative_points: Number of points to sample from each negative
                (background) mask. Default: 5.
            postprocessor: Post-processor applied after predict().
                Defaults to :func:`~instantlearn.components.postprocessing.default_postprocessor`
                (MaskIoMNMS + BoxIoMNMS).
        """
        if postprocessor is None:
            postprocessor = default_postprocessor()
        super().__init__(postprocessor=postprocessor)

        self.device = device
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution
        self.precision = precision
        self.compile_models = compile_models
        self.model_id = model_id
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

        # Negative prompt support
        self.negative_mask_converter = NegativeMaskToPoints(num_points_per_mask=num_negative_points)
        self._negative_points: torch.Tensor | None = None
        self._negative_geometry_features: torch.Tensor | None = None
        self._negative_geometry_mask: torch.Tensor | None = None

        # Preprocessors and postprocessor
        self.image_preprocessor = Sam3Preprocessor(target_size=resolution).to(device)
        self.prompt_preprocessor = Sam3PromptPreprocessor(target_size=resolution).to(device)
        self.sam3_postprocessor = Sam3Postprocessor(
            target_size=resolution,
            threshold=confidence_threshold,
            mask_threshold=0.5,
        ).to(device)

        # Tokenizer for text prompts (still from transformers, but not used in ONNX path)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_id)

        self.model = (
            Sam3Model.from_pretrained(
                model_id,
                torch_dtype=precision_to_torch_dtype(precision),
            )
            .to(device)
            .eval()
        )

    # -- Public API --

    @torch.inference_mode()
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
        """
        reference_batch = Batch.collate(reference)
        self._extract_negative_points(reference_batch)

        if self.prompt_mode == Sam3PromptMode.CLASSIC:
            self._fit_classic(reference_batch)
        else:
            self._fit_visual_exemplar(reference_batch)

    @torch.inference_mode()
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
        """
        if self.prompt_mode == Sam3PromptMode.VISUAL_EXEMPLAR:
            return self.apply_postprocessing(self._predict_visual_exemplar(target))
        return self.apply_postprocessing(self._predict_classic(target))

    # -- Fit internals --

    def _extract_negative_points(self, reference_batch: Batch) -> None:
        """Extract negative points from background masks in reference samples.

        Scans all samples for masks with category_id == BACKGROUND_CATEGORY_ID,
        converts them to point prompts, and normalizes coordinates to [0, 1].
        Points are cached in ``self._negative_points`` for use in predict.

        Args:
            reference_batch: Batch of reference samples.
        """
        self._negative_points = None
        all_neg_points: list[torch.Tensor] = []

        for sample in reference_batch.samples:
            if sample.masks is None or sample.category_ids is None:
                continue

            for mask, cat_id in zip(sample.masks, sample.category_ids, strict=True):
                if int(cat_id) != BACKGROUND_CATEGORY_ID:
                    continue
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
                if not mask_t.any():
                    continue
                neg_pts, _ = self.negative_mask_converter(mask_t.unsqueeze(0))
                if neg_pts.numel() == 0:
                    continue
                # Normalize to [0, 1] relative to mask spatial dims (x / w, y / h)
                img_h, img_w = mask_t.shape[-2:]
                neg_pts_norm = neg_pts.float()
                neg_pts_norm[:, 0] = neg_pts_norm[:, 0] / img_w
                neg_pts_norm[:, 1] = neg_pts_norm[:, 1] / img_h
                all_neg_points.append(neg_pts_norm)

        if all_neg_points:
            self._negative_points = torch.cat(all_neg_points, dim=0).to(self.device)
            logger.info("Cached %d negative points from background masks", self._negative_points.shape[0])

    def _fit_classic(self, reference_batch: Batch) -> None:
        """Store category mapping from reference batch.

        Args:
            reference_batch: Batch of reference samples.
        """
        self.category_mapping = self._build_category_mapping(reference_batch)
        self._pre_encode_negative_geometry(reference_batch)

    def _pre_encode_negative_geometry(self, reference_batch: Batch) -> None:
        """Pre-encode negative points using reference image vision features.

        Encodes the cached negative points against the reference image so that
        cross-image transfer uses visual content from the reference rather than
        spatial coordinates that may not transfer to different target images.

        Args:
            reference_batch: Batch of reference samples with images.
        """
        self._negative_geometry_features = None
        self._negative_geometry_mask = None

        if self._negative_points is None or self._negative_points.numel() == 0:
            return

        # Find the first reference sample with an image
        ref_sample = next((s for s in reference_batch.samples if s.image is not None), None)
        if ref_sample is None:
            return

        image_tensor = ref_sample.image.unsqueeze(0) if ref_sample.image.ndim == 3 else ref_sample.image
        pixel_values, _ = self.image_preprocessor(image_tensor.to(self.device))
        vision_embeds = self.model.get_vision_features(pixel_values)
        fpn_hidden_states = vision_embeds["fpn_hidden_states"][:-1]
        fpn_position_encoding = vision_embeds["fpn_position_encoding"][:-1]

        neg_pts = self._negative_points.unsqueeze(0).to(dtype=fpn_hidden_states[0].dtype)
        num_neg = neg_pts.shape[1]
        neg_labels = torch.zeros((1, num_neg), dtype=torch.long, device=self.device)
        neg_mask = torch.ones(1, num_neg, dtype=torch.bool, device=self.device)

        geometry_outputs = self.model.geometry_encoder(
            point_embeddings=neg_pts,
            point_mask=neg_mask,
            point_labels=neg_labels,
            img_feats=fpn_hidden_states,
            img_pos_embeds=fpn_position_encoding,
            drop_spatial_bias=True,
        )

        self._negative_geometry_features = geometry_outputs["last_hidden_state"]
        self._negative_geometry_mask = geometry_outputs["attention_mask"]
        logger.info(
            "Pre-encoded %d negative points into geometry features [%s]",
            num_neg,
            self._negative_geometry_features.shape,
        )

    def _fit_visual_exemplar(self, reference_batch: Batch) -> None:
        """Encode visual exemplar features from reference images and boxes/points.

        Supports n-shot encoding: multiple prompts for the same category are
        batched together in a single geometry encoder call, enabling self-attention
        between shots from the same image. Features from different images for the
        same category are concatenated, giving the DETR decoder richer conditioning.

        This means passing 3 point prompts for "shoe" produces a single, stronger
        exemplar rather than 3 separate weak ones.

        Note:
            Both boxes and points are encoded as point-only because point encoding
            transfers better across images (0.83 vs 0.59 mIoU): grid_sample at a
            single point captures local appearance without averaging over an ROI
            region.

        Args:
            reference_batch: Batch of reference samples with images and bboxes/points.

        Raises:
            ValueError: If no reference samples contain bboxes or points.
        """
        encoded_by_category, category_text_map = self._encode_batch_prompts(reference_batch)

        if not encoded_by_category:
            msg = "VISUAL_EXEMPLAR mode requires at least one reference sample with bboxes or points."
            raise ValueError(msg)

        geometry_features, geometry_masks, category_ids, text_prompts = self._aggregate_category_features(
            encoded_by_category,
            category_text_map,
        )
        self.exemplar_geometry_features = geometry_features
        self.exemplar_geometry_mask = geometry_masks
        self.exemplar_category_ids = category_ids
        self.exemplar_text_features, self.exemplar_text_mask = self._cache_text_features(text_prompts)
        self.category_mapping = self._build_category_mapping(reference_batch)

        # Log shot counts per category
        shot_info = {
            category_text_map[cat_id]: sum(f[0].shape[1] for f in encoded_by_category[cat_id])
            for cat_id in self.exemplar_category_ids
        }
        logger.info(
            "Cached %d category exemplar(s) with n-shot encoding: %s, category_ids=%s",
            len(self.exemplar_category_ids),
            shot_info,
            self.exemplar_category_ids,
        )

    def _encode_batch_prompts(
        self,
        reference_batch: Batch,
    ) -> tuple[dict[int, list[tuple[torch.Tensor, torch.Tensor]]], dict[int, str]]:
        """Encode all samples' box/point prompts into per-category geometry features.

        Iterates over every sample in the batch so that the same category can
        accumulate geometry features from different reference images (cross-image
        n-shot).  For example, two samples each containing a "shoe" box will both
        append to ``encoded_by_category[shoe_id]``; the downstream aggregation step
        concatenates those features to give the DETR decoder richer conditioning.

        Args:
            reference_batch: Batch of reference samples with images and bboxes/points.

        Returns:
            Tuple of (encoded_by_category, category_text_map) where
            encoded_by_category maps cat_id to list of (features, mask) tuples
            and category_text_map maps cat_id to text name.
        """
        encoded_by_category: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
        category_text_map: dict[int, str] = {}

        for sample in reference_batch.samples:
            self._encode_sample_prompts(sample, encoded_by_category, category_text_map)

        return encoded_by_category, category_text_map

    def _encode_sample_prompts(
        self,
        sample: Sample,
        encoded_by_category: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
        category_text_map: dict[int, str],
    ) -> None:
        """Encode one sample's box/point prompts into per-category geometry features.

        Args:
            sample: Reference sample with image and bboxes/points.
            encoded_by_category: Accumulator mapping cat_id to encoded features.
            category_text_map: Accumulator mapping cat_id to text name.

        Raises:
            ValueError: If the sample has prompts but no image.
        """
        bboxes = sample.bboxes
        points = sample.points
        has_bboxes = bboxes is not None and not (isinstance(bboxes, (np.ndarray, torch.Tensor)) and bboxes.size == 0)
        has_points = points is not None and not (isinstance(points, (np.ndarray, torch.Tensor)) and points.size == 0)

        if not has_bboxes and not has_points:
            return
        if sample.image is None:
            msg = "VISUAL_EXEMPLAR mode requires images in reference samples."
            raise ValueError(msg)

        # Extract vision features
        image_tensor = sample.image.unsqueeze(0) if sample.image.ndim == 3 else sample.image
        pixel_values, original_sizes = self.image_preprocessor(image_tensor.to(self.device))
        vision_embeds = self.model.get_vision_features(pixel_values)
        fpn_hidden_states = vision_embeds["fpn_hidden_states"][:-1]
        fpn_position_encoding = vision_embeds["fpn_position_encoding"][:-1]

        # Build aligned metadata lists
        num_prompts = max(len(bboxes) if has_bboxes else 0, len(points) if has_points else 0)
        categories = sample.categories if sample.categories is not None else ["visual"] * num_prompts
        category_ids = sample.category_ids if sample.category_ids is not None else [0] * num_prompts

        # Convert prompts to point coords grouped by category
        category_coords: dict[int, list[torch.Tensor]] = defaultdict(list)
        prompts = bboxes if has_bboxes else points

        for prompt, category, cat_id in zip(prompts, categories, category_ids, strict=True):
            cat_id_int = int(cat_id)
            if cat_id_int == BACKGROUND_CATEGORY_ID:
                continue  # handled via _extract_negative_points
            if has_bboxes:
                input_boxes, _ = self.prompt_preprocessor(original_sizes, input_boxes=prompt)
                coord = input_boxes[..., :2]  # box center (1, 1, 2)
            else:
                _, coord = self.prompt_preprocessor(original_sizes, input_points=prompt)
            category_coords[cat_id_int].append(coord)
            category_text_map[cat_id_int] = category

        # Prepare negative points (already normalized to [0,1] during _extract_negative_points)
        neg_coords = None
        if self._negative_points is not None and self._negative_points.numel() > 0:
            neg_coords = self._negative_points.unsqueeze(0)  # [1, M, 2]

        # Encode each category's points together (same-image n-shot batching)
        for cat_id, coords_list in category_coords.items():
            all_coords = torch.cat(coords_list, dim=1)  # [1, N, 2]
            num_positive = all_coords.shape[1]

            # Append negative points so they participate in geometry self-attention
            if neg_coords is not None:
                all_coords = torch.cat([all_coords, neg_coords.to(all_coords)], dim=1)
            total_points = all_coords.shape[1]

            labels = torch.ones((1, total_points), dtype=torch.long, device=self.device)
            labels[:, num_positive:] = 0  # label=0 for negative points

            geometry_outputs = self.model.geometry_encoder(
                point_embeddings=all_coords.to(dtype=fpn_hidden_states[0].dtype),
                point_mask=torch.ones(1, total_points, dtype=torch.bool, device=self.device),
                point_labels=labels,
                img_feats=fpn_hidden_states,
                img_pos_embeds=fpn_position_encoding,
                drop_spatial_bias=self.drop_spatial_bias,
            )
            encoded_by_category[cat_id].append((
                geometry_outputs["last_hidden_state"],
                geometry_outputs["attention_mask"],
            ))

    @staticmethod
    def _aggregate_category_features(
        encoded_by_category: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
        category_text_map: dict[int, str],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int], list[str]]:
        """Merge per-category geometry features across reference images.

        Args:
            encoded_by_category: Per-category encoded features.
            category_text_map: Mapping from cat_id to text name.

        Returns:
            Tuple of (geometry_features, geometry_masks, category_ids, text_prompts).
        """
        all_geometry_features: list[torch.Tensor] = []
        all_geometry_masks: list[torch.Tensor] = []
        all_category_ids: list[int] = []
        all_text_prompts: list[str] = []

        for cat_id in sorted(encoded_by_category.keys()):
            features_list = encoded_by_category[cat_id]
            if len(features_list) == 1:
                geo_feats, geo_mask = features_list[0]
            else:
                geo_feats = torch.cat([f[0] for f in features_list], dim=1)
                geo_mask = torch.cat([f[1] for f in features_list], dim=1)

            all_geometry_features.append(geo_feats)
            all_geometry_masks.append(geo_mask)
            all_category_ids.append(cat_id)
            all_text_prompts.append(category_text_map[cat_id])

        return all_geometry_features, all_geometry_masks, all_category_ids, all_text_prompts

    def _cache_text_features(
        self,
        text_prompts: list[str],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Tokenize and embed text prompts.

        Args:
            text_prompts: Text prompts aligned with exemplar categories.

        Returns:
            Tuple of (text_features, text_masks) per exemplar.
        """
        unique_prompts = list(dict.fromkeys(text_prompts))
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

        return (
            [text_cache[p][0] for p in text_prompts],
            [text_cache[p][1] for p in text_prompts],
        )

    # -- Predict internals --

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
                text_inputs = self.tokenizer(
                    [text or "visual"],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=32,
                )
                input_ids = text_inputs.input_ids.to(self.device)
                attention_mask = text_inputs.attention_mask.to(self.device)

                # Prepare box inputs if bbox is provided (xyxy format)
                input_boxes = None
                input_boxes_labels = None
                if bbox is not None and len(bbox):
                    input_boxes, _ = self.prompt_preprocessor(original_sizes, input_boxes=bbox)
                    input_boxes_labels = torch.ones((1, 1), dtype=torch.long, device=self.device)

                # Prepare point inputs if point is provided (xy format)
                input_points = None
                input_points_labels = None
                if point is not None and len(point):
                    _, input_points = self.prompt_preprocessor(original_sizes, input_points=point)
                    input_points_labels = torch.ones((1, 1), dtype=torch.long, device=self.device)

                # Build precomputed geometry features from positive prompts + cached negatives
                precomputed_geo = None
                precomputed_geo_mask = None

                # Encode positive points/boxes on the target image (if any)
                has_positive_geo = (input_boxes is not None and input_boxes.numel() > 0) or (
                    input_points is not None and input_points.numel() > 0
                )
                if has_positive_geo:
                    fpn_hidden_states = vision_embeds["fpn_hidden_states"][:-1]
                    fpn_position_encoding = vision_embeds["fpn_position_encoding"][:-1]
                    dtype = fpn_hidden_states[0].dtype

                    geo_kwargs: dict = {
                        "img_feats": fpn_hidden_states,
                        "img_pos_embeds": fpn_position_encoding,
                    }
                    if input_points is not None and input_points.numel() > 0:
                        geo_kwargs["point_embeddings"] = input_points.to(dtype=dtype)
                        geo_kwargs["point_mask"] = torch.ones(
                            1,
                            input_points.shape[1],
                            dtype=torch.bool,
                            device=self.device,
                        )
                        geo_kwargs["point_labels"] = input_points_labels
                    if input_boxes is not None and input_boxes.numel() > 0:
                        geo_kwargs["box_embeddings"] = input_boxes.to(dtype=dtype)
                        geo_kwargs["box_mask"] = torch.ones(
                            1,
                            input_boxes.shape[1],
                            dtype=torch.bool,
                            device=self.device,
                        )
                        geo_kwargs["box_labels"] = input_boxes_labels

                    geo_out = self.model.geometry_encoder(**geo_kwargs)
                    precomputed_geo = geo_out["last_hidden_state"]
                    precomputed_geo_mask = geo_out["attention_mask"]

                # Append cached negative geometry features (pre-encoded on reference image)
                if self._negative_geometry_features is not None:
                    neg_geo = self._negative_geometry_features
                    neg_mask = self._negative_geometry_mask
                    if precomputed_geo is not None:
                        precomputed_geo = torch.cat([precomputed_geo, neg_geo], dim=1)
                        precomputed_geo_mask = torch.cat([precomputed_geo_mask, neg_mask], dim=1)
                    else:
                        precomputed_geo = neg_geo
                        precomputed_geo_mask = neg_mask

                with torch.no_grad():
                    outputs = self.model(
                        vision_embeds=vision_embeds,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        precomputed_geometry_features=precomputed_geo,
                        precomputed_geometry_mask=precomputed_geo_mask,
                    )

                # Postprocess
                result = self.sam3_postprocessor(outputs, target_sizes=[img_size])
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
                pixel_values, _ = self.image_preprocessor(image_tensor.to(self.device))
                vision_embeds = self.model.get_vision_features(pixel_values)

            all_masks: list[torch.Tensor] = []
            all_boxes: list[torch.Tensor] = []
            all_labels: list[torch.Tensor] = []

            # Run detection for each cached exemplar
            for geo_feats, geo_mask, text_feats, text_mask, cat_id in zip(
                self.exemplar_geometry_features,
                self.exemplar_geometry_mask,
                self.exemplar_text_features,
                self.exemplar_text_mask,
                self.exemplar_category_ids,
                strict=True,
            ):
                outputs = self.model(
                    vision_embeds=vision_embeds,
                    text_embeds=text_feats,
                    attention_mask=text_mask.long(),
                    precomputed_geometry_features=geo_feats,
                    precomputed_geometry_mask=geo_mask,
                )

                # Postprocess
                result = self.sam3_postprocessor(outputs, target_sizes=[img_size])
                boxes_with_scores = torch.cat(
                    [result[0]["boxes"], result[0]["scores"].unsqueeze(1)],
                    dim=1,
                )
                all_masks.append(result[0]["masks"].cpu())
                all_boxes.append(boxes_with_scores.cpu())
                all_labels.append(torch.full((len(result[0]["boxes"]),), cat_id, dtype=torch.int64))

            results.append(self._aggregate_results(all_masks, all_boxes, all_labels, img_size))

        return results

    # -- Utilities --

    @staticmethod
    def _build_category_mapping(reference_batch: Batch) -> dict[str, int]:
        """Build category name → id mapping from reference samples.

        Args:
            reference_batch: Batch of reference samples.

        Returns:
            Mapping from category name to category id.
        """
        mapping: dict[str, int] = {}
        for sample in reference_batch.samples:
            if sample.categories is None or sample.category_ids is None:
                continue
            for category_id, category in zip(sample.category_ids, sample.categories, strict=False):
                if int(category_id) == BACKGROUND_CATEGORY_ID:
                    continue
                if category not in mapping:
                    mapping[category] = int(category_id)
        return mapping

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
