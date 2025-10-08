# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This Pipeline uses a zero-shot object detector (from Huggingface) to generate boxes for SAM."""

from getiprompt.filters.masks import BoxAwareMaskFilter, MaskFilter
from getiprompt.filters.priors import MultiInstancePriorFilter
from getiprompt.models.models import load_sam_model
from getiprompt.pipelines.pipeline_base import Pipeline
from getiprompt.processes.mask_processors import MaskProcessor, MasksToPolygons
from getiprompt.processes.prompt_generators import GroundedObjectDetector, GroundingModel
from getiprompt.processes.segmenters import SamDecoder, Segmenter
from getiprompt.types import Image, Priors, Results, Text
from getiprompt.utils.constants import SAMModelName
from getiprompt.utils.decorators import track_duration


class GroundedSAM(Pipeline):
    """This Pipeline uses a zero-shot object detector (from Huggingface) to generate boxes for SAM."""

    def __init__(
        self,
        sam: SAMModelName = SAMModelName.SAM_HQ_TINY,
        grounding_model: GroundingModel = GroundingModel.LLMDET_TINY,
        precision: str = "bf16",
        compile_models: bool = False,
        benchmark_inference_speed: bool = False,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
        device: str = "cuda",
        image_size: int | tuple[int, int] | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            sam: The SAM model name.
            grounding_model: The grounding model to use.
            precision: The precision to use for the model.
            compile_models: Whether to compile the models.
            benchmark_inference_speed: Whether to benchmark the inference speed.
            box_threshold: The box threshold.
            text_threshold: The text threshold.
            device: The device to use.
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
        self.prompt_generator: GroundedObjectDetector = GroundedObjectDetector(
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            template=GroundedObjectDetector.Template.specific_object,
            grounding_model=grounding_model,
            precision=precision,
            compile_models=compile_models,
            benchmark_inference_speed=benchmark_inference_speed,
        )
        self.segmenter: Segmenter = SamDecoder(
            sam_predictor=self.sam_predictor,
        )
        self.box_aware_mask_filter: MaskFilter = BoxAwareMaskFilter()
        self.multi_instance_prior_filter: MultiInstancePriorFilter = MultiInstancePriorFilter()
        self.mask_processor: MaskProcessor = MasksToPolygons()
        self.text_priors: Text | None = None

    @track_duration
    def learn(self, reference_images: list[Image], reference_priors: list[Priors]) -> Results:  # noqa: ARG002
        """Perform learning step on the reference images and priors."""
        if not all(p.text is not None for p in reference_priors):
            msg = "reference_priors must have all text types"
            raise ValueError(msg)
        # If all priors are the same use only the first one, else use all.
        if not all(p.text.data for p in reference_priors):
            msg = "Different image-level text priors not supported."
            raise ValueError(msg)

        self.text_priors = reference_priors[0].text

    @track_duration
    def infer(self, target_images: list[Image]) -> Results:
        """Perform inference step on the target images."""
        # Start running the pipeline
        target_images = self.resize_images(target_images)
        priors = self.prompt_generator(target_images, [self.text_priors] * len(target_images))
        priors = self.multi_instance_prior_filter(priors)
        masks, _, used_boxes = self.segmenter(target_images, priors)
        annotations = self.mask_processor(masks)

        # write output
        results = Results()
        results.priors = priors
        results.used_boxes = used_boxes
        results.masks = masks
        results.annotations = annotations
        return results
