# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Matcher model, based on the paper 'Segment Anything with One Shot Using All-Purpose Feature Matching'."""

from typing import TYPE_CHECKING

from getiprompt.components import MaskAdder, MasksToPolygons, SamDecoder
from getiprompt.components.encoders import ImageEncoder
from getiprompt.components.feature_extractors.local_feature_extractor import LocalFeatureExtractor
from getiprompt.components.feature_selectors import AllFeaturesSelector, FeatureSelector
from getiprompt.components.filters import PointFilter
from getiprompt.components.prompt_generators import BidirectionalPromptGenerator
from getiprompt.data.base.batch import Batch
from getiprompt.types import Results
from getiprompt.utils.benchmark import track_duration
from getiprompt.utils.constants import SAMModelName

from .base import Model
from .foundation import load_sam_model

if TYPE_CHECKING:
    from getiprompt.components.prompt_generators.base import PromptGenerator


class Matcher(Model):
    """This is the Matcher model.

    It's based on the paper "[ICLR'24] Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching"
    https://arxiv.org/abs/2305.13310.

    Main novelties:
    - Uses DinoV2 patch encoding instead of SAM for encoding the images, resulting in a more robust feature extractor
    - Uses a bidirectional prompt generator to generate prompts for the segmenter
    - Has a more complex mask postprocessing step to remove and merge masks

    Note that the post processing mask filtering techniques are different from that of the original paper.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> from getiprompt.models import Matcher
        >>> from torchvision import tv_tensors
        >>> from getiprompt.types import Priors, Results
        >>>
        >>> matcher = Matcher()
        >>>
        >>> # Create mock inputs
        >>> ref_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        >>> target_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        >>> ref_priors = Priors()
        >>> ref_priors.masks.add(torch.ones(30, 30, dtype=torch.bool), class_id=1)
        >>>
        >>> # Run learn and infer
        >>> learn_results = matcher.learn([tv_tensors.Image(ref_image)], [ref_priors])
        >>> infer_results = matcher.infer([tv_tensors.Image(target_image)])
        >>>
        >>> isinstance(learn_results, Results) and isinstance(infer_results, Results)
        True
        >>> infer_results.masks is not None and infer_results.annotations is not None
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
        benchmark_inference_speed: bool = False,
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
            benchmark_inference_speed: Whether to benchmark the inference speed.
            device: The device to use for the model.
        """
        super().__init__()
        self.sam_predictor = load_sam_model(
            sam,
            device,
            precision=precision,
            compile_models=compile_models,
            benchmark_inference_speed=benchmark_inference_speed,
        )
        self.encoder: ImageEncoder = ImageEncoder(
            model_id=encoder_model,
            device=device,
            precision=precision,
            compile_models=compile_models,
            benchmark_inference_speed=benchmark_inference_speed,
        )
        # Local feature extraction with mask pooling
        self.local_feature_extractor = LocalFeatureExtractor(
            input_size=self.encoder.input_size,
            patch_size=self.encoder.patch_size,
            device=device,
        )
        self.feature_selector: FeatureSelector = AllFeaturesSelector()
        self.prompt_generator: PromptGenerator = BidirectionalPromptGenerator(
            encoder_input_size=self.encoder.input_size,
            encoder_patch_size=self.encoder.patch_size,
            encoder_feature_size=self.encoder.feature_size,
            num_background_points=num_background_points,
        )
        self.point_filter = PointFilter(num_foreground_points=num_foreground_points)
        self.segmenter: SamDecoder = SamDecoder(
            sam_predictor=self.sam_predictor,
            mask_similarity_threshold=mask_similarity_threshold,
        )
        self.mask_adder = MaskAdder(segmenter=self.segmenter)
        self.mask_processor = MasksToPolygons()
        self.reference_features = None
        self.reference_masks = None

    @track_duration
    def learn(self, reference_batch: Batch) -> None:
        """Perform learning step on the reference images and priors."""
        # Encode reference images to batched tensor
        reference_embeddings = self.encoder(images=reference_batch.images)
        # Extract local features and pooled masks
        reference_features, self.reference_masks = self.local_feature_extractor(
            reference_embeddings,
            reference_batch.masks,
            reference_batch.category_ids,
            reference_batch.is_reference,
        )
        self.reference_features = self.feature_selector(reference_features)

    @track_duration
    def infer(self, target_batch: Batch) -> Results:
        """Perform inference step on the target images."""
        target_embeddings = self.encoder(images=target_batch.images)
        point_prompts, similarities_per_image = self.prompt_generator(
            self.reference_features,
            self.reference_masks,
            target_embeddings,
            target_batch.images,
        )
        point_prompts = self.point_filter(point_prompts)
        masks, used_points, _ = self.segmenter(
            target_batch.images,
            point_prompts=point_prompts,
            similarities=similarities_per_image,
        )
        annotations = self.mask_processor(masks)

        # write output
        results = Results()
        results.priors = point_prompts
        results.used_points = used_points
        results.masks = masks
        results.annotations = annotations
        results.similarities = similarities_per_image
        return results
