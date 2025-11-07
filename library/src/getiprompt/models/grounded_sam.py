# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This model uses a zero-shot object detector (from Huggingface) to generate boxes for SAM."""

from getiprompt.components import SamDecoder
from getiprompt.components.filters import BoxPromptFilter
from getiprompt.components.prompt_generators import GroundingModel, TextToBoxPromptGenerator
from getiprompt.data.base.batch import Batch
from getiprompt.types import Results
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
        self.segmenter: SamDecoder = SamDecoder(sam_predictor=self.sam_predictor)
        self.prompt_filter: BoxPromptFilter = BoxPromptFilter()

    @track_duration
    def learn(self, reference_batch: Batch) -> None:
        """Perform learning step on the reference images and priors.

        Args:
            reference_batch(Batch): The reference batch.
        """
        self.category_mapping = {}
        for sample in reference_batch.samples:
            for category_id, category in zip(sample.category_ids, sample.categories, strict=False):
                if category not in self.category_mapping:
                    self.category_mapping[category] = int(category_id)

    @track_duration
    def infer(self, target_batch: Batch) -> Results:
        """Perform inference step on the target images.

        Args:
            target_batch(Batch): The target batch.

        Returns:
            Results: The results.
        """
        # Start running the model
        box_prompts = self.prompt_generator(target_batch.images, self.category_mapping)
        box_prompts = self.prompt_filter(box_prompts)
        masks, _, used_boxes = self.segmenter(
            target_batch.images,
            box_prompts=box_prompts,
        )

        # write output
        results = Results()
        results.box_prompts = box_prompts
        results.used_boxes = used_boxes
        results.masks = masks
        return results
