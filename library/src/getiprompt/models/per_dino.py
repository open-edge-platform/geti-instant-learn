# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PerDino model."""

from typing import TYPE_CHECKING

from getiprompt.components import CosineSimilarity, SamDecoder
from getiprompt.components.encoders import ImageEncoder
from getiprompt.components.feature_extractors import MaskedFeatureExtractor
from getiprompt.components.filters import ClassOverlapMaskFilter, PointPromptFilter
from getiprompt.components.prompt_generators import GridPromptGenerator
from getiprompt.data.base.batch import Batch
from getiprompt.types import Results
from getiprompt.utils.benchmark import track_duration
from getiprompt.utils.constants import SAMModelName

from .base import Model
from .foundation import load_sam_model

if TYPE_CHECKING:
    from getiprompt.components.prompt_generators.base import PromptGenerator


class PerDino(Model):
    """This is the PerDino algorithm model.

    It matches reference objects to target images by comparing their features extracted by Dino
    and using Cosine Similarity. A grid prompt generator is used to generate prompts for the
    segmenter and to allow for multi object target images.

    Examples:
        >>> from getiprompt.models import PerDino
        >>> from getiprompt.data.base import Batch
        >>> from getiprompt.data.base.sample import Sample
        >>> from getiprompt.types import Results
        >>> import torch
        >>> import numpy as np

        >>> perdino = PerDino()

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
        >>> perdino.learn(ref_batch)
        >>> infer_results = perdino.infer(target_batch)

        >>> isinstance(infer_results, Results)
        True
        >>> infer_results.masks is not None
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
        # Local feature extraction with mask pooling
        self.masked_feature_extractor = MaskedFeatureExtractor(
            input_size=self.encoder.input_size,
            patch_size=self.encoder.patch_size,
            device=device,
        )
        self.similarity_matcher = CosineSimilarity()
        self.prompt_generator: PromptGenerator = GridPromptGenerator(
            num_grid_cells=num_grid_cells,
            similarity_threshold=similarity_threshold,
            num_bg_points=num_background_points,
        )
        self.prompt_filter = PointPromptFilter(num_foreground_points=num_foreground_points)
        self.segmenter: SamDecoder = SamDecoder(
            sam_predictor=self.sam_predictor,
            mask_similarity_threshold=mask_similarity_threshold,
        )
        self.class_overlap_mask_filter = ClassOverlapMaskFilter()
        self.reference_embeddings = None

    @track_duration
    def learn(self, reference_batch: Batch) -> None:
        """Perform learning step on the reference images and priors."""
        # Start running the model
        reference_embeddings = self.encoder(reference_batch.images)
        self.masked_ref_embeddings, _ = self.masked_feature_extractor(
            reference_embeddings,
            reference_batch.masks,
            reference_batch.category_ids,
        )

    @track_duration
    def infer(self, target_batch: Batch) -> Results:
        """Perform inference step on the target images."""
        # Start running the model
        target_images = target_batch.images
        image_sizes = [image.shape[-2:] for image in target_images]
        target_embeddings = self.encoder(target_images)
        similarities = self.similarity_matcher(self.masked_ref_embeddings, target_embeddings, image_sizes)
        point_prompts = self.prompt_generator(similarities, target_images)
        point_prompts = self.prompt_filter(point_prompts)
        masks, used_points, _ = self.segmenter(
            target_images,
            point_prompts=point_prompts,
            similarities=similarities,
        )
        masks, used_points = self.class_overlap_mask_filter(masks, used_points)

        # write output
        results = Results()
        results.point_prompts = point_prompts
        results.used_points = used_points
        results.masks = masks
        results.similarities = similarities
        return results
