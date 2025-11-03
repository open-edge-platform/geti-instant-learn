# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This model uses a zero-shot object detector (from Huggingface) to generate boxes for SAM."""

from getiprompt.components import MasksToPolygons, SamDecoder
from getiprompt.components.filters import MultiInstancePriorFilter
from getiprompt.components.prompt_generators import GroundingModel, TextToBoxPromptGenerator
from getiprompt.data.base.batch import Batch
from getiprompt.types import Results, Text
from getiprompt.utils.benchmark import track_duration
from getiprompt.utils.constants import SAMModelName

from .base import Model
from .foundation import load_sam_model


class GroundedSAM(Model):
    """This model uses a zero-shot object detector (from Huggingface) to generate boxes for SAM."""

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
    ) -> None:
        """Initialize the model.

        Args:
            sam: The SAM model name.
            grounding_model: The grounding model to use.
            precision: The precision to use for the model.
            compile_models: Whether to compile the models.
            benchmark_inference_speed: Whether to benchmark the inference speed.
            box_threshold: The box threshold.
            text_threshold: The text threshold.
            device: The device to use.
        """
        super().__init__()
        self.sam_predictor = load_sam_model(
            sam,
            device,
            precision=precision,
            compile_models=compile_models,
            benchmark_inference_speed=benchmark_inference_speed,
        )
        self.prompt_generator: TextToBoxPromptGenerator = TextToBoxPromptGenerator(
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            template=TextToBoxPromptGenerator.Template.specific_object,
            model_id=grounding_model,
            precision=precision,
            compile_models=compile_models,
            benchmark_inference_speed=benchmark_inference_speed,
        )
        self.segmenter: SamDecoder = SamDecoder(
            sam_predictor=self.sam_predictor,
        )
        self.multi_instance_prior_filter: MultiInstancePriorFilter = MultiInstancePriorFilter()
        self.mask_processor = MasksToPolygons()
        self.text_priors: Text | None = None

    @track_duration
    def learn(self, reference_batch: Batch) -> None:
        """Perform learning step on the reference images and priors.

        Args:
            reference_batch(Batch): The reference batch.

        Raises:
            ValueError: If the reference priors do not have all text types.
        """
        self.text_priors = list(
            dict.fromkeys(cat for categories in reference_batch.categories for cat in categories),
        )

    @track_duration
    def infer(self, target_batch: Batch) -> Results:
        """Perform inference step on the target images.

        Args:
            target_batch(Batch): The target batch.

        Returns:
            Results: The results.
        """
        # Start running the model
        pred_samples = self.prompt_generator(target_batch.images, self.text_priors)
        box_prompts = self.multi_instance_prior_filter(pred_samples)
        masks, _, used_boxes = self.segmenter(
            target_batch.images,
            box_prompts=box_prompts,
        )
        annotations = self.mask_processor(masks)

        # write output
        results = Results()
        results.priors = box_prompts
        results.used_boxes = used_boxes
        results.masks = masks
        results.annotations = annotations
        return results
