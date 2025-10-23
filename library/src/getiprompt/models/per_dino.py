# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PerDino model."""

from typing import TYPE_CHECKING

from getiprompt.components import CosineSimilarity, MaskAdder, MasksToPolygons, SamDecoder
from getiprompt.components.encoders import ImageEncoder
from getiprompt.components.feature_selectors import AverageFeatures, FeatureSelector
from getiprompt.components.filters import ClassOverlapMaskFilter, MaxPointFilter
from getiprompt.components.prompt_generators import GridPromptGenerator
from getiprompt.models import Model, load_sam_model
from getiprompt.types import Image, Priors, Results
from getiprompt.utils.benchmark import track_duration
from getiprompt.utils.constants import SAMModelName

if TYPE_CHECKING:
    from getiprompt.components.prompt_generators.base import PromptGenerator


class PerDino(Model):
    """This is the PerDino algorithm model.

    It matches reference objects to target images by comparing their features extracted by Dino
    and using Cosine Similarity. A grid prompt generator is used to generate prompts for the
    segmenter and to allow for multi object target images.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> from getiprompt.models import PerDino
        >>> from getiprompt.types import Image, Priors, Results
        >>>
        >>> perdino = PerDino()
        >>>
        >>> # Create mock inputs
        >>> ref_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        >>> target_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        >>> ref_priors = Priors()
        >>> ref_priors.masks.add(torch.ones(30, 30, dtype=torch.bool), class_id=1)
        >>>
        >>> # Run learn and infer
        >>> learn_results = perdino.learn([Image(ref_image)], [ref_priors])
        >>> infer_results = perdino.infer([Image(target_image)])
        >>>
        >>> isinstance(learn_results, Results) and isinstance(infer_results, Results)
        True
        >>> infer_results.masks is not None
        True
        >>> infer_results.annotations is not None
        True
    """

    def __init__(
        self,
        sam: SAMModelName = SAMModelName.SAM_HQ_TINY,
        encoder_model: str = "dinov3_large",
        num_foreground_points: int = 40,
        num_background_points: int = 2,
        num_grid_cells: int = 16,
        similarity_threshold: float = 0.65,
        mask_similarity_threshold: float | None = 0.42,
        precision: str = "bf16",
        compile_models: bool = False,
        benchmark_inference_speed: bool = False,
        device: str = "cuda",
    ) -> None:
        """Initialize the PerDino model.

        Args:
            sam: The name of the SAM model to use.
            num_foreground_points: The number of foreground points to use.
            num_background_points: The number of background points to use.
            num_grid_cells: The number of grid cells to use.
            similarity_threshold: The similarity threshold for the similarity matcher.
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
        self.feature_selector: FeatureSelector = AverageFeatures()
        self.similarity_matcher = CosineSimilarity()
        self.prompt_generator: PromptGenerator = GridPromptGenerator(
            num_grid_cells=num_grid_cells,
            similarity_threshold=similarity_threshold,
            num_bg_points=num_background_points,
        )
        self.point_filter = MaxPointFilter(
            max_num_points=num_foreground_points,
        )
        self.segmenter: SamDecoder = SamDecoder(
            sam_predictor=self.sam_predictor,
            mask_similarity_threshold=mask_similarity_threshold,
        )
        self.prior_mask_from_points = MaskAdder(segmenter=self.segmenter)
        self.mask_processor = MasksToPolygons()
        self.class_overlap_mask_filter = ClassOverlapMaskFilter()
        self.reference_features = None

    @track_duration
    def learn(self, reference_images: list[Image], reference_priors: list[Priors]) -> Results:
        """Perform learning step on the reference images and priors."""
        reference_priors = self.prior_mask_from_points(reference_images, reference_priors)

        # Start running the model
        reference_features, _ = self.encoder(
            reference_images,
            reference_priors,
        )
        self.reference_features = self.feature_selector(reference_features)

    @track_duration
    def infer(self, target_images: list[Image]) -> Results:
        """Perform inference step on the target images."""
        # Start running the model
        target_features, _ = self.encoder(target_images)
        similarities = self.similarity_matcher(self.reference_features, target_features, target_images)
        priors = self.prompt_generator(similarities, target_images)
        priors = self.point_filter(priors)
        masks, used_points, _ = self.segmenter(target_images, priors, similarities)
        masks, used_points = self.class_overlap_mask_filter(masks, used_points)
        annotations = self.mask_processor(masks)

        # write output
        results = Results()
        results.priors = priors
        results.used_points = used_points
        results.masks = masks
        results.annotations = annotations
        results.similarities = similarities
        return results
