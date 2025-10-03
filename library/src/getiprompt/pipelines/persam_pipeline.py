# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PerSam pipeline."""

from typing import TYPE_CHECKING

from getiprompt.filters.masks import ClassOverlapMaskFilter, MaskFilter
from getiprompt.filters.priors import MaxPointFilter, PriorFilter, PriorMaskFromPoints
from getiprompt.models.models import load_sam_model
from getiprompt.pipelines.pipeline_base import Pipeline
from getiprompt.processes.encoders import Encoder, SamEncoder
from getiprompt.processes.feature_selectors import AverageFeatures, FeatureSelector
from getiprompt.processes.mask_processors import MaskProcessor, MasksToPolygons
from getiprompt.processes.prompt_generators import GridPromptGenerator
from getiprompt.processes.segmenters import SamDecoder, Segmenter
from getiprompt.processes.similarity_matchers import CosineSimilarity, SimilarityMatcher
from getiprompt.types import Image, Priors, Results
from getiprompt.utils.constants import SAMModelName
from getiprompt.utils.decorators import track_duration

if TYPE_CHECKING:
    from getiprompt.filters.priors.prior_filter_base import PriorFilter
    from getiprompt.processes.prompt_generators.prompt_generator_base import PromptGenerator


class PerSam(Pipeline):
    """This is the PerSam algorithm pipeline.

    It's based on the paper "Personalize Segment Anything Model with One Shot"
    https://arxiv.org/abs/2305.03048.

    It matches reference objects to target images by comparing their features extracted by SAM
    and using Cosine Similarity. A grid prompt generator is used to generate prompts for the
    segmenter and to allow for multi object target images.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> from getiprompt.pipelines import PerSam
        >>> from getiprompt.types import Image, Priors, Results
        >>>
        >>> persam = PerSam()
        >>>
        >>> # Create mock inputs
        >>> ref_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        >>> target_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        >>> ref_priors = Priors()
        >>> ref_priors.masks.add(torch.ones(30, 30, dtype=torch.bool), class_id=1)
        >>>
        >>> # Run learn and infer
        >>> learn_results = persam.learn([Image(ref_image)], [ref_priors])
        >>> infer_results = persam.infer([Image(target_image)])
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
        num_grid_cells: int = 16,
        similarity_threshold: float = 0.65,
        mask_similarity_threshold: float | None = 0.42,
        precision: str = "bf16",
        compile_models: bool = False,
        benchmark_inference_speed: bool = False,
        device: str = "cuda",
        image_size: int | tuple[int, int] | None = None,
    ) -> None:
        """Initialize the PerSam pipeline.

        Args:
            sam: The name of the SAM model to use.
            num_foreground_points: The number of foreground points to use.
            num_background_points: The number of background points to use.
            num_grid_cells: The number of grid cells to use.
            similarity_threshold: The similarity threshold for the similarity matcher.
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
        self.encoder: Encoder = SamEncoder(sam_predictor=self.sam_predictor)
        self.feature_selector: FeatureSelector = AverageFeatures()
        self.similarity_matcher: SimilarityMatcher = CosineSimilarity()
        self.prompt_generator: PromptGenerator = GridPromptGenerator(
            num_grid_cells=num_grid_cells,
            similarity_threshold=similarity_threshold,
            num_bg_points=num_background_points,
        )
        self.point_filter: PriorFilter = MaxPointFilter(
            max_num_points=num_foreground_points,
        )
        self.segmenter: Segmenter = SamDecoder(
            sam_predictor=self.sam_predictor,
            mask_similarity_threshold=mask_similarity_threshold,
        )
        self.mask_processor: MaskProcessor = MasksToPolygons()
        self.class_overlap_mask_filter: MaskFilter = ClassOverlapMaskFilter()
        self.reference_features = None
        self.prior_mask_from_points: PriorFilter = PriorMaskFromPoints(segmenter=self.segmenter)

    @track_duration
    def learn(self, reference_images: list[Image], reference_priors: list[Priors]) -> Results:
        """Perform learning step on the reference images and priors."""
        reference_images = self.resize_images(reference_images)
        reference_priors = self.prior_mask_from_points(reference_images, reference_priors)
        reference_priors = self.resize_masks(reference_priors)

        # Start running the pipeline
        reference_features, _ = self.encoder(
            reference_images,
            reference_priors,
        )
        self.reference_features = self.feature_selector(reference_features)

    @track_duration
    def infer(self, target_images: list[Image]) -> Results:
        """Perform inference step on the target images."""
        target_images = self.resize_images(target_images)

        # Start running the pipeline
        target_features, _ = self.encoder(target_images)
        similarities = self.similarity_matcher(
            self.reference_features,
            target_features,
            target_images,
        )
        priors = self.prompt_generator(similarities, target_images)
        priors = self.point_filter(priors)
        masks, used_points, _ = self.segmenter(target_images, priors)
        masks = self.class_overlap_mask_filter(masks, used_points)
        annotations = self.mask_processor(masks)

        # write output
        results = Results()
        results.priors = priors
        results.used_points = used_points
        results.masks = masks
        results.annotations = annotations
        results.similarities = similarities
        return results
