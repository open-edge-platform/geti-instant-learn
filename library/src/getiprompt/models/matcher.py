# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Matcher model, based on the paper 'Segment Anything with One Shot Using All-Purpose Feature Matching'."""

import torch

from getiprompt.components import SamDecoder
from getiprompt.components.encoders.timm import TimmImageEncoder
from getiprompt.components.feature_extractors import MaskedFeatureExtractor
from getiprompt.components.filters import PointPromptFilter
from getiprompt.components.prompt_generators import BidirectionalPromptGenerator
from getiprompt.data.base.batch import Batch
from getiprompt.utils.constants import SAMModelName

from .base import Model
from .foundation import load_sam_model


class Matcher(Model):
    """This is the Matcher model.

    It's based on the paper "[ICLR'24] Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching"
    https://arxiv.org/abs/2305.13310.

    Main novelties:
    - Uses patch encoding instead of SAM for encoding the images, resulting in a more robust feature extractor
    - Uses a bidirectional prompt generator to generate prompts for the segmenter
    - Has a more complex mask postprocessing step to remove and merge masks

    Note that the post processing mask filtering techniques are different from that of the original paper.

    Examples:
        >>> from getiprompt.models import Matcher
        >>> from getiprompt.data.base import Batch
        >>> from getiprompt.data.base.sample import Sample
        >>> from getiprompt.types import Results
        >>> import torch
        >>> import numpy as np

        >>> matcher = Matcher()

        >>> # Create mock inputs
        >>> ref_image = torch.zeros((3, 1024, 1024))
        >>> target_image = torch.zeros((3, 1024, 1024))
        >>> ref_mask = torch.ones(30, 30, dtype=torch.bool)

        >>> # Create reference sample
        >>> ref_sample = Sample(
        ...     image=ref_image,
        ...     masks=ref_mask.unsqueeze(0),
        ...     category_ids=np.array([1]),
        ...     is_reference=[True],
        ...     categories=["object"],
        ... )
        >>> ref_batch = Batch.collate([ref_sample])

        >>> # Create target sample
        >>> target_sample = Sample(
        ...     image=target_image,
        ...     is_reference=[False],
        ...     categories=["object"],
        ... )
        >>> target_batch = Batch.collate([target_sample])

        >>> # Run learn and infer
        >>> matcher.learn(ref_batch)
        >>> infer_results = matcher.infer(target_batch)

        >>> isinstance(infer_results, Results)
        True

        >>> infer_results.masks is not None
        True
    """

    def __init__(
        self,
        sam: SAMModelName = SAMModelName.SAM_HQ_TINY,
        num_foreground_points: int = 40,
        num_background_points: int = 2,
        encoder_model: str = "dinov3_large",
        mask_similarity_threshold: float | None = 0.38,
        precision: str = "bf16",
        compile_models: bool = False,
        device: str = "cuda",
    ) -> None:
        """Initialize the Matcher model.

        Args:
            sam: The name of the SAM model to use.
            num_foreground_points: The number of foreground points to use.
            num_background_points: The number of background points to use.
            mask_similarity_threshold: The similarity threshold for the mask.
            encoder_model: ImageEncoder model ID to use.
            precision: The precision to use for the model.
            compile_models: Whether to compile the models.
            device: The device to use for the model.
        """
        super().__init__()
        self.sam_predictor = load_sam_model(
            sam,
            device,
            precision=precision,
            compile_models=compile_models,
        )
        self.encoder = TimmImageEncoder(
            model_id=encoder_model,
            device=device,
            precision=precision,
            compile_models=compile_models,
        )
        # Local feature extraction with mask pooling
        self.masked_feature_extractor = MaskedFeatureExtractor(
            input_size=self.encoder.input_size,
            patch_size=self.encoder.patch_size,
            device=device,
        )
        self.prompt_generator = BidirectionalPromptGenerator(
            encoder_input_size=self.encoder.input_size,
            encoder_patch_size=self.encoder.patch_size,
            encoder_feature_size=self.encoder.feature_size,
            num_background_points=num_background_points,
        )
        self.prompt_filter = PointPromptFilter(num_foreground_points=num_foreground_points)
        self.segmenter: SamDecoder = SamDecoder(
            sam_predictor=self.sam_predictor,
            mask_similarity_threshold=mask_similarity_threshold,
        )
        self.masked_ref_embeddings = None
        self.ref_masks = None

    def learn(self, reference_batch: Batch) -> None:
        """Perform learning step on the reference images and priors."""
        # Encode reference images to batched tensor
        self.ref_embeddings = self.encoder(images=reference_batch.images)
        # Extract local features and pooled masks
        self.masked_ref_embeddings, self.ref_masks = self.masked_feature_extractor(
            self.ref_embeddings,
            reference_batch.masks,
            reference_batch.category_ids,
        )

    def infer(self, target_batch: Batch) -> list[dict[str, torch.Tensor]]:
        """Perform inference step on the target images.

        Args:
            target_batch(Batch): The target batch.

        Returns:
            predictions(list[dict[str, torch.Tensor]]): A list of predictions.
            Each prediction contains:
                "pred_masks": torch.Tensor of shape [num_masks, H, W]
                "pred_points": torch.Tensor of shape [num_points, 4] with last dimension [x, y, score, fg_label]
                "pred_boxes": torch.Tensor of shape [num_boxes, 5] with last dimension [x1, y1, x2, y2, score]
                "pred_labels": torch.Tensor of shape [num_masks]
        """
        target_images = target_batch.images
        original_sizes = [image.shape[-2:] for image in target_images]
        target_embeddings = self.encoder(images=target_batch.images)
        point_prompts, similarities_per_image = self.prompt_generator(
            self.ref_embeddings,
            self.masked_ref_embeddings,
            self.ref_masks,
            target_embeddings,
            original_sizes,
        )
        point_prompts = self.prompt_filter(point_prompts)
        return self.segmenter(
            target_images,
            point_prompts=point_prompts,
            similarities=similarities_per_image,
        )
