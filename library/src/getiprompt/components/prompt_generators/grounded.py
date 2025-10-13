"""Generate bounding boxes using a zero shot object detector."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import numpy as np
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from getiprompt.components.prompt_generators.base import PromptGenerator
from getiprompt.foundation.model_optimizer import optimize_model
from getiprompt.types import Boxes, Image, Priors, Text
from getiprompt.utils import precision_to_torch_dtype


class GroundingModel(Enum):
    """The model to use for the grounding."""

    GROUNDING_DINO_BASE = "IDEA-Research/grounding-dino-base"
    GROUNDING_DINO_TINY = "IDEA-Research/grounding-dino-tiny"
    LLMDET_TINY = "fushh7/llmdet_swin_tiny_hf"
    LLMDET_BASE = "fushh7/llmdet_swin_base_hf"
    LLMDET_LARGE = "fushh7/llmdet_swin_large_hf"


class GroundedObjectDetector(PromptGenerator):
    """This class generates prompts for the segmenter.

    This class uses a zero-shot object detector from HuggingFace to produce bounding box priors.
    """

    class Template:
        """Template for object prompts."""

        specific_object = "{prior}"
        all_objects = "an object"
        group_of_objects = "{prior} (group)"

    def __init__(
        self,
        box_threshold: float,
        text_threshold: float,
        template: str,
        grounding_model: GroundingModel = GroundingModel.LLMDET_TINY,
        device: str = "cuda",
        precision: str = "bf16",
        compile_models: bool = False,
        benchmark_inference_speed: bool = False,
    ) -> None:
        """Initialize the GroundedObjectDetector.

        Args:
            box_threshold: The box threshold.
            text_threshold: The text threshold.
            template: The template to use for the prompt
            grounding_model: The grounding model to use.
            device: The device to use.
            precision: The precision to use for the model.
            compile_models: Whether to compile the models.
            benchmark_inference_speed: Whether to benchmark the inference speed.
        """
        super().__init__()
        self.model_id = grounding_model.value
        self.device = device
        self.model, self.processor = self._load_grounding_model_and_processor(
            self.model_id, precision, device, compile_models, benchmark_inference_speed
        )
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.template = template

    def forward(
        self,
        target_images: list[Image],
        text_priors: list[Text],
    ) -> list[Priors]:
        """This generates bounding box prompt candidates (or priors) based on the text priors.

        Args:
            target_images: The target images
            text_priors: List[Text] the priors to use as an input

        Returns:
            List[Priors] List of priors, one per target image instance
        """
        if len(text_priors) != len(target_images):
            msg = "Need one text prior per image"
            raise ValueError(msg)
        # convert to Pillow images
        pil_images = [image.to_pil() for image in target_images]
        sizes = [image.size[::-1] for image in pil_images]

        # convert to strings seperated by a '.'
        text_labels = []
        for text_prior in text_priors:
            text = [self.template.format(prior=text_prior.get(cid)) for cid in text_prior.class_ids()]
            text_labels.append(". ".join(text) + ".")

        # Run the grounding model
        inputs = self.processor(images=pil_images, text=text_labels, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=self.model.dtype):
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=sizes,
        )

        # Generate all priors from the result of the Dino model.
        priors_per_image: list[Priors] = []
        for result, text_prior in zip(results, text_priors, strict=False):
            result_per_label = {}  # store all boxes of a certain class
            # This returns the xyxy box, float score and text label.
            for box, score, label_text in zip(result["boxes"], result["scores"], result["labels"], strict=False):
                # fuzzy match the label text to the class name
                label_id = text_prior.find_best(label_text)
                row = np.array([*[float(b) for b in box], float(score), float(label_id)])
                if label_id in result_per_label:
                    result_per_label[label_id].append(row)
                else:
                    result_per_label[label_id] = [row]
            boxes = Boxes()
            # Remap the label text back to the label id (class id) and create a Boxes instance
            for label_id, boxes_result in result_per_label.items():
                boxes.add(data=np.array(boxes_result), class_id=int(label_id))
            # Wrap the Boxes into a Priors class
            priors = Priors(boxes=boxes)
            priors_per_image.append(priors)
        return priors_per_image

    @staticmethod
    def _load_grounding_model_and_processor(
        model_id: str, precision: str, device: str, compile_models: bool, benchmark_inference_speed: bool
    ) -> tuple[AutoModelForZeroShotObjectDetection, AutoProcessor]:
        """Load the grounding model and processor.

        Args:
            model_id: The model id to load.
            precision: The precision to use for the model.
            device: The device to use for the model.
            compile_models: Whether to compile the models.
            benchmark_inference_speed: Whether to benchmark the inference speed.
        """
        processor = AutoProcessor.from_pretrained(model_id)
        if model_id.startswith("fushh7/llmdet_swin"):
            # LLMDET has a slightly different interface, use lazy import for efficiency
            from getiprompt.foundation.grounding_dino import GroundingDinoForObjectDetection

            model = GroundingDinoForObjectDetection.from_pretrained(
                model_id, torch_dtype=precision_to_torch_dtype(precision)
            )
        else:
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id, torch_dtype=precision_to_torch_dtype(precision)
            )
        model = optimize_model(
            model=model.to(device).eval(),
            device=device,
            precision=precision_to_torch_dtype(precision),
            compile_models=compile_models,
            benchmark_inference_speed=benchmark_inference_speed,
        )
        return model, processor
