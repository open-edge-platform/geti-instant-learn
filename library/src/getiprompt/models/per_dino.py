# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PerDino model."""

import torch

from getiprompt.components import CosineSimilarity, SamDecoder
from getiprompt.components.encoders import ImageEncoder
from getiprompt.components.feature_extractors import MaskedFeatureExtractor, ReferenceFeatures
from getiprompt.components.prompt_generators import GridPromptGenerator
from getiprompt.components.sam import load_sam_model
from getiprompt.data.base.batch import Batch
from getiprompt.data.base.sample import Sample
from getiprompt.models.base import Model
from getiprompt.utils.constants import Backend, SAMModelName


class PerDino(Model):
    """PerDino algorithm model for one-shot segmentation.

    Matches reference objects to target images by comparing features extracted by DINOv2
    using cosine similarity. A grid prompt generator creates multi-object aware prompts.

    The pipeline is fully traceable (ONNX/TorchScript compatible):
    - Encoder → MaskedFeatureExtractor → CosineSimilarity → GridPromptGenerator → SamDecoder

    Examples:
        >>> from getiprompt.models import PerDino
        >>> from getiprompt.data.base import Batch
        >>> from getiprompt.data.base.sample import Sample
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
        use_nms: bool = True,
        precision: str = "bf16",
        compile_models: bool = False,
        device: str = "cuda",
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
            use_nms: Whether to use NMS in SamDecoder.
            precision: Model precision ("bf16", "fp32").
            compile_models: Whether to compile models with torch.compile.
            device: Device for inference.
        """
        super().__init__()
        self.sam_predictor = load_sam_model(
            sam,
            device=device,
            precision=precision,
            compile_models=compile_models,
        )

        self.encoder = ImageEncoder(
            model_id=encoder_model,
            backend=Backend.TIMM,
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
            use_nms=use_nms,
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

    def predict(self, target: Sample | list[Sample] | Batch) -> list[dict[str, torch.Tensor]]:
        """Predict masks for target images.

        Args:
            target: Target data to infer. Accepts:
                - Sample: A single target sample
                - list[Sample]: A list of target samples
                - Batch: A batch of target samples

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
        return self.segmenter(
            target_batch.images,
            self.ref_features.category_ids,
            point_prompts=point_prompts,
            similarities=similarities,
        )
