# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Matcher model, based on the paper 'Segment Anything with One Shot Using All-Purpose Feature Matching'."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

from getiprompt.components.encoders import ImageEncoder
from getiprompt.components.feature_extractors import MaskedFeatureExtractor, ReferenceFeatures
from getiprompt.components.prompt_generators import BidirectionalPromptGenerator
from getiprompt.components.sam import SamDecoder, load_sam_model
from getiprompt.data.base.batch import Batch
from getiprompt.models.base import Model
from getiprompt.utils.constants import Backend, SAMModelName


class EncoderForwardFeaturesWrapper(nn.Module):
    """Wrapper for image encoder to expose forward_features method for export."""

    IMAGENET_DEFAULT_MEAN = torch.tensor((0.485, 0.456, 0.406))
    IMAGENET_DEFAULT_STD = torch.tensor((0.229, 0.224, 0.225))

    def __init__(
        self,
        encoder: nn.Module,
        ignore_token_length: int,
        input_size: int = 512,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.ignore_token_length = ignore_token_length
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get encoder features."""
        x = x.float() / 255.0
        x = functional.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear")
        x = (x - self.IMAGENET_DEFAULT_MEAN[None, :, None, None]) / self.IMAGENET_DEFAULT_STD[None, :, None, None]
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
    ) -> None:
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
        """Single image forward pass for export: target_image [1, 3, H, W] → (masks, scores, labels)."""
        # Get original size from input tensor [1, 3, H, W]
        original_sizes = torch.stack([
            torch.tensor(target_image.size(2)),
            torch.tensor(target_image.size(3)),
        ]).unsqueeze(0)

        # Encode target [1, num_patches, embed_dim]
        target_embeddings = self.encoder(target_image)

        # Generate prompts using frozen ref_features
        # point_prompts: [1, C, max_points, 4], num_points: [1, C], similarities: [1, C, feat_size, feat_size]
        point_prompts, similarities = self.prompt_generator.forward(
            self.ref_embeddings,
            self.masked_ref_embeddings,
            self.flatten_ref_masks,
            self.category_ids,
            target_embeddings,
            original_sizes,
        )

        # Decode using export-friendly method (single image, returns tensors)
        return self.sam_decoder.forward_export(
            target_image[0],  # Single image [3, H, W]
            self.category_ids,
            point_prompts[0],  # [C, max_points, 4]
            similarities[0],  # [C, feat_size, feat_size]
        )


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
        self.sam_predictor = load_sam_model(
            sam,
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

    def fit(self, reference_batch: Batch) -> ReferenceFeatures:
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
        return self.ref_features

    def predict(self, target_batch: Batch) -> list[dict[str, torch.Tensor]]:
        """Predict masks for target images.

        Args:
            target_batch: Batch containing target images.

        Returns:
            List of predictions per image, each containing:
                "pred_masks": [num_masks, H, W]
                "pred_scores": [num_masks]
                "pred_labels": [num_masks] - category IDs

        Raises:
            RuntimeError: If fit() has not been called before predict().
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
        point_prompts, similarities = self.prompt_generator(
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
            similarities=similarities,
        )

    @torch.no_grad()
    def export(
        self,
        reference_features: ReferenceFeatures,
        export_dir: str | Path = Path("./exports/matcher"),
        backend: Backend = Backend.ONNX,
    ) -> Path:
        """Export model components.

        Args:
            reference_features: Reference features obtained from fit().
            export_dir: Directory to save exported models.
            backend: Export backend (ONNX, OpenVINO).
            **kwargs: Additional export parameters.

        Returns:
            Path to export directory.

        Raises:
            ImportError: If OpenVINO is selected but not installed.
        """
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        matcher = MatcherInferenceGraph(
            encoder=EncoderForwardFeaturesWrapper(
                self.encoder._model.model,
                ignore_token_length=self.encoder._model.ignore_token_length,
            ),
            prompt_generator=self.prompt_generator,
            sam_decoder=self.segmenter,
            ref_features=reference_features,
        )

        target_image = torch.randn(1, 3, self.encoder.input_size, self.encoder.input_size)
        if backend == Backend.ONNX:
            onnx_path = export_path / "matcher.onnx"
            torch.onnx.export(
                matcher,
                args=(target_image,),
                f=onnx_path,
                input_names=["target_image"],
                output_names=["masks", "scores", "labels"],
                dynamic_axes={
                    "target_image": {2: "height", 3: "width"},
                    "masks": {0: "num_masks", 1: "height", 2: "width"},
                    "scores": {0: "num_masks"},
                    "labels": {0: "num_masks"},
                },
                verbose=True,
            )
            return onnx_path

        if backend == Backend.OPENVINO:
            try:
                import openvino

                # Export to ONNX first, then convert to OpenVINO
                # Direct PyTorch → OpenVINO conversion fails on many ops (aten::pad, aten::unbind, etc.)
                # ONNX → OpenVINO conversion has much better support
                onnx_path = export_path / "matcher.onnx"
                torch.onnx.export(
                    matcher,
                    args=(target_image,),
                    f=onnx_path,
                    input_names=["target_image"],
                    output_names=["masks", "scores", "labels"],
                    dynamic_axes={
                        "target_image": {2: "height", 3: "width"},
                        "masks": {0: "num_masks", 1: "height", 2: "width"},
                        "scores": {0: "num_masks"},
                        "labels": {0: "num_masks"},
                    },
                )
                # Convert ONNX to OpenVINO
                ov_model = openvino.convert_model(onnx_path)
                openvino.save_model(ov_model, export_path / "matcher.xml")
                return export_path / "matcher.xml"
            except ImportError:
                msg = "OpenVINO is not installed. Please install it to use OpenVINO export."
                raise ImportError(msg)

        return export_path
