# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Matcher model, based on the paper 'Segment Anything with One Shot Using All-Purpose Feature Matching'."""

from pathlib import Path

import torch

from getiprompt.components.encoders import ImageEncoder
from getiprompt.components.feature_extractors import MaskedFeatureExtractor, ReferenceFeatures
from getiprompt.components.mask_decoder import SamDecoder
from getiprompt.components.prompt_generators import BidirectionalPromptGenerator
from getiprompt.components.sam.base import SAMPredictor
from getiprompt.data.base.batch import Batch
from getiprompt.models.base import Model
from getiprompt.utils.constants import Backend, SAMModelName
from torch import nn
from torch.nn import functional


class EncoderForwardFeaturesWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, ignore_token_length: int):
        super().__init__()
        self.encoder = encoder
        self.ignore_token_length = ignore_token_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder.forward_features(x)
        features = features[:, self.ignore_token_length :, :]  # ignore CLS and other tokens
        return functional.normalize(features, p=2, dim=-1)


class MatcherInferenceGraph(nn.Module):
    """Traceable inference graph with frozen reference features."""
    
    def __init__(self, encoder, prompt_generator, sam_decoder, ref_features: ReferenceFeatures):
        super().__init__()
        self.encoder = encoder
        self.prompt_generator = prompt_generator
        self.sam_decoder = sam_decoder
        
        # Freeze reference features as model constants
        self.register_buffer("ref_embeddings", ref_features.ref_embeddings)
        self.register_buffer("masked_ref_embeddings", ref_features.masked_ref_embeddings)
        self.register_buffer("flatten_ref_masks", ref_features.flatten_ref_masks)
        self.register_buffer("category_ids", torch.tensor(ref_features.category_ids))
    
    def forward(self, target_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single forward pass: target_image → (masks, scores, labels)."""
        # Encode target
        # Get original sizes [T, 2]
        original_sizes = torch.tensor(
            [image.size()[-2:] for image in target_image.images],
            device=self.ref_embeddings.device,
        )

        target_embeddings = self.encoder(target_image)
        
        # Generate prompts using frozen ref_features
        point_prompts, num_points, similarities = self.prompt_generator(
            self.ref_embeddings,
            self.masked_ref_embeddings,
            self.flatten_ref_masks,
            self.category_ids,
            target_embeddings,
            original_sizes,
        )
        
        # Decode
        return self.sam_decoder(target_image, point_prompts, num_points, similarities)




class Matcher(Model):
    """Matcher model for one-shot segmentation.

    Based on "[ICLR'24] Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching"
    https://arxiv.org/abs/2305.13310.

    The pipeline is fully traceable (ONNX/TorchScript compatible):
    - Encoder → MaskedFeatureExtractor → BidirectionalPromptGenerator → TraceableSamDecoder

    Examples:
        >>> from getiprompt.models import Matcher
        >>> from getiprompt.data.base import Batch
        >>> from getiprompt.data.base.sample import Sample
        >>> import torch
        >>> import numpy as np

        >>> matcher = Matcher()

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
        >>> matcher.fit(ref_batch)
        >>> predict_results = matcher.predict(target_batch)

        >>> isinstance(predict_results, Results)
        True

        >>> predict_results.masks is not None
        True
    """

    def __init__(
        self,
        sam: SAMModelName = SAMModelName.SAM_HQ_TINY,
        num_foreground_points: int = 40,
        num_background_points: int = 2,
        encoder_model: str = "dinov3_large",
        mask_similarity_threshold: float | None = 0.38,
        use_mask_refinement: bool = True,
        precision: str = "bf16",
        compile_models: bool = False,
        device: str = "cuda",
    ) -> None:
        """Initialize the Matcher model.

        Args:
            sam: SAM model variant to use.
            num_foreground_points: Maximum foreground points per category.
            num_background_points: Background points per category.
            encoder_model: Image encoder model ID.
            mask_similarity_threshold: Threshold for similarity-based mask filtering.
            use_mask_refinement: Whether to use 2-stage mask refinement with box prompts.
            precision: Model precision ("bf16", "fp32").
            compile_models: Whether to compile models with torch.compile.
            device: Device for inference.
        """
        super().__init__()
        # SAM predictor
        self.sam_predictor = SAMPredictor(
            sam,
            backend=Backend.PYTORCH,
            device=device,
            precision=precision,
            compile_models=compile_models,
        )

        # Image encoder
        self.encoder = ImageEncoder(
            model_id=encoder_model,
            backend=Backend.TIMM,
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
        )

        # SAM decoder
        self.segmenter = SamDecoder(
            sam_predictor=self.sam_predictor,
            mask_similarity_threshold=mask_similarity_threshold,
            use_mask_refinement=use_mask_refinement,
        )

        # Reference features (set during fit)
        self.ref_features: ReferenceFeatures | None = None

    def fit(self, reference_batch: Batch) -> None:
        """Learn from reference images.

        Args:
            reference_batch: Batch containing reference images, masks, and category IDs.
        """
        ref_embeddings = self.encoder(images=reference_batch.images)
        self.ref_features = self.masked_feature_extractor(
            ref_embeddings,
            reference_batch.masks,
            reference_batch.category_ids,
        )

    def predict(self, target_batch: Batch) -> list[dict[str, torch.Tensor]]:
        """Predict masks for target images.

        Args:
            target_batch: Batch containing target images.

        Returns:
            List of predictions per image, each containing:
                "pred_masks": [num_masks, H, W]
                "pred_scores": [num_masks]
                "pred_labels": [num_masks] - category IDs
        """
        if self.ref_features is None:
            msg = "No reference features. Call fit() first."
            raise RuntimeError(msg)

        # Get original sizes [T, 2]
        original_sizes = torch.tensor(
            [image.size()[-2:] for image in target_batch.images],
            device=self.ref_features.device,
        )

        # Encode all targets [T, num_patches, embed_dim]
        target_embeddings = self.encoder(images=target_batch.images)

        # Generate prompts [T, C, max_points, 4], [T, C], [T, C, feat_size, feat_size]
        point_prompts, num_points, similarities = self.prompt_generator(
            self.ref_features.ref_embeddings,
            self.ref_features.masked_ref_embeddings,
            self.ref_features.flatten_ref_masks,
            self.ref_features.category_ids,
            target_embeddings,
            original_sizes,
        )

        # Decode masks for all images
        return self.segmenter(
            target_batch.images,
            self.ref_features.category_ids,
            point_prompts=point_prompts,
            num_points=num_points,
            similarities=similarities,
        )

    def export(
        self,
        reference_features: ReferenceFeatures,
        export_dir: str | Path = Path("./exports/matcher"),
        backend: Backend = Backend.ONNX,
    ) -> Path:
        """Export model components.

        Args:
            export_dir: Directory to save exported models.
            backend: Export backend (ONNX, OpenVINO).
            **kwargs: Additional export parameters.

        Returns:
            Path to export directory.
        """
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        matcher = MatcherInferenceGraph(
            encoder=EncoderForwardFeaturesWrapper(self.encoder, ignore_token_length=1),
            prompt_generator=self.prompt_generator,
            sam_decoder=self.segmenter.sam_decoder,
            ref_features=reference_features,
        )
        

        return export_path
