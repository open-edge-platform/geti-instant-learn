# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Matcher pipeline, based on the paper 'Segment Anything with One Shot Using All-Purpose Feature Matching'."""

from typing import TYPE_CHECKING

from getiprompt.filters.priors import MaxPointFilter, PriorFilter, PriorMaskFromPoints
from getiprompt.models.models import load_sam_model
from getiprompt.pipelines.pipeline_base import Pipeline
from getiprompt.processes.encoders import DinoEncoder, Encoder
from getiprompt.processes.feature_selectors import AllFeaturesSelector, FeatureSelector
from getiprompt.processes.mask_processors import MaskProcessor, MasksToPolygons
from getiprompt.processes.prompt_generators import BidirectionalPromptGenerator
from getiprompt.processes.segmenters import SamDecoder, Segmenter
from getiprompt.types import Image, Priors, Results
from getiprompt.utils.constants import SAMModelName
from getiprompt.utils.decorators import track_duration

if TYPE_CHECKING:
    from getiprompt.filters.priors.prior_filter_base import PriorFilter
    from getiprompt.processes.prompt_generators.prompt_generator_base import PromptGenerator


class Matcher(Pipeline):
    """This is the Matcher pipeline.

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
        >>> from getiprompt.pipelines import Matcher
        >>> from getiprompt.types import Image, Priors, Results
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
        >>> learn_results = matcher.learn([Image(ref_image)], [ref_priors])
        >>> infer_results = matcher.infer([Image(target_image)])
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
        mask_similarity_threshold: float | None = 0.38,
        precision: str = "bf16",
        compile_models: bool = False,
        benchmark_inference_speed: bool = False,
        device: str = "cuda",
        image_size: int | tuple[int, int] | None = None,
    ) -> None:
        """Initialize the Matcher pipeline.

        Args:
            sam: The name of the SAM model to use.
            num_foreground_points: The number of foreground points to use.
            num_background_points: The number of background points to use.
            mask_similarity_threshold: The similarity threshold for the mask.
            precision: The precision to use for the model.
            compile_models: Whether to compile the models.
            benchmark_inference_speed: Whether to benchmark the inference speed.
            device: The device to use for the model.
            image_size: The size of the image to use, if None, the image will not be resized.
        """
        super().__init__(image_size=image_size)
        self.sam_predictor = load_sam_model(
            sam,
            device,
            precision=precision,
            compile_models=compile_models,
            benchmark_inference_speed=benchmark_inference_speed,
        )
        self.encoder: Encoder = DinoEncoder(
            precision=precision,
            compile_models=compile_models,
            benchmark_inference_speed=benchmark_inference_speed,
            device=device,
        )
        self.feature_selector: FeatureSelector = AllFeaturesSelector()
        self.prompt_generator: PromptGenerator = BidirectionalPromptGenerator(
            encoder_input_size=self.encoder.encoder_input_size,
            encoder_patch_size=self.encoder.patch_size,
            encoder_feature_size=self.encoder.feature_size,
            num_background_points=num_background_points,
        )
        self.point_filter: PriorFilter = MaxPointFilter(max_num_points=num_foreground_points)
        self.segmenter: Segmenter = SamDecoder(
            sam_predictor=self.sam_predictor,
            mask_similarity_threshold=mask_similarity_threshold,
        )
        self.prior_mask_from_points: PriorFilter = PriorMaskFromPoints(segmenter=self.segmenter)
        self.mask_processor: MaskProcessor = MasksToPolygons()
        self.reference_features = None
        self.reference_masks = None

    @track_duration
    def learn(self, reference_images: list[Image], reference_priors: list[Priors]) -> Results:
        """Perform learning step on the reference images and priors."""
        reference_images = self.resize_images(reference_images)
        reference_priors = self.prior_mask_from_points(reference_images, reference_priors)
        reference_priors = self.resize_masks(reference_priors)

        # Start running the pipeline
        reference_features, self.reference_masks = self.encoder(reference_images, reference_priors)
        self.reference_features = self.feature_selector(reference_features)

    @track_duration
    def infer(self, target_images: list[Image]) -> Results:
        """Perform inference step on the target images."""
        target_images = self.resize_images(target_images)

        # Start running the pipeline
        target_features, _ = self.encoder(target_images)
        priors, similarities = self.prompt_generator(
            self.reference_features,
            target_features,
            self.reference_masks,
            target_images,
        )
        priors = self.point_filter(priors)
        masks, used_points, _ = self.segmenter(target_images, priors, similarities)
        annotations = self.mask_processor(masks)

        # write output
        results = Results()
        results.priors = priors
        results.used_points = used_points
        results.masks = masks
        results.annotations = annotations
        results.similarities = similarities
        return results
