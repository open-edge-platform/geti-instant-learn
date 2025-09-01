# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This is a Pipeline based on grounding Dino with a SAM decoder."""


from getiprompt.filters.masks import BoxAwareMaskFilter, ClassOverlapMaskFilter, MaskFilter
from getiprompt.filters.priors import MultiInstancePriorFilter
from getiprompt.models.models import load_sam_model
from getiprompt.pipelines.pipeline_base import Pipeline
from getiprompt.processes.mask_processors import MaskProcessor, MasksToPolygons
from getiprompt.processes.prompt_generators import GroundingDinoBoxGenerator
from getiprompt.processes.segmenters import SamDecoder, Segmenter
from getiprompt.types import Image, Priors, Results, Text
from getiprompt.utils.constants import SAMModelName
from getiprompt.utils.decorators import track_duration


class GroundingDinoSAM(Pipeline):
    """This Pipeline uses GroundingDino (from Huggingface) to generate boxes for SAM."""

    def __init__(
        self,
        sam: SAMModelName = SAMModelName.SAM_HQ_TINY,
        apply_mask_refinement: bool = True,
        precision: str = "bf16",
        compile_models: bool = False,
        verbose: bool = False,
        backbone_size: GroundingDinoBoxGenerator.Size = GroundingDinoBoxGenerator.Size.TINY,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
        device: str = "cuda",
        image_size: int | tuple[int, int] | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            sam: The SAM model name.
            apply_mask_refinement: Whether to apply mask refinement.
            precision: The precision to use for the model.
            compile_models: Whether to compile the models.
            verbose: Whether to print verbose output of the model optimization process.
            backbone_size: The size of the backbone.
            box_threshold: The box threshold.
            text_threshold: The text threshold.
            device: The device to use.
            image_size: The size of the image to use, if None, the image will not be resized.
        """
        super().__init__(image_size=image_size)
        self.sam_predictor = load_sam_model(
            sam, device, precision=precision, compile_models=compile_models, verbose=verbose
        )
        self.prompt_generator: GroundingDinoBoxGenerator = GroundingDinoBoxGenerator(
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            size=backbone_size,
            precision=precision,
            compile_models=compile_models,
            verbose=verbose,
            template=GroundingDinoBoxGenerator.Template.specific_object,
        )
        self.segmenter: Segmenter = SamDecoder(
            sam_predictor=self.sam_predictor,
            apply_mask_refinement=apply_mask_refinement,
            skip_points_in_existing_masks=False,  # not relevant for boxes
        )
        self.box_aware_mask_filter: MaskFilter = BoxAwareMaskFilter()
        self.multi_instance_prior_filter: MultiInstancePriorFilter = MultiInstancePriorFilter()
        self.mask_processor: MaskProcessor = MasksToPolygons()
        self.class_overlap_mask_filter: MaskFilter = ClassOverlapMaskFilter()
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
