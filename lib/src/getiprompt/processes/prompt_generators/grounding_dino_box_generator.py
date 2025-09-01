# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import numpy as np
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from getiprompt.models.model_optimizer import optimize_model
from getiprompt.processes.prompt_generators.prompt_generator_base import PromptGenerator
from getiprompt.types import Boxes, Image, Priors, Text
from getiprompt.utils import precision_to_torch_dtype


class GroundingDinoBoxGenerator(PromptGenerator):
    """This class generates prompts for the segmenter.

    This class uses the GroundingDino from HuggingFace to produce bounding box priors.
    """

    class Size(Enum):
        """Size of the backbone."""

        BASE = "base"
        TINY = "tiny"

    class Template:
        """Template for object prompts."""

        specific_object = "{prior}"
        all_objects = "an object"

    def __init__(
        self,
        box_threshold: float,
        text_threshold: float,
        size: Size | str,
        template: str,
        device: str = "cuda",
        precision: str = "bf16",
        compile_models: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize the GroundingDinoBoxGenerator.

        Args:
            box_threshold: The box threshold.
            text_threshold: The text threshold.
            size: The size of the model.
            template: The template to use for the prompt
            device: The device to use.
            precision: The precision to use for the model.
            compile_models: Whether to compile the models.
            verbose: Whether to print verbose output of the model optimization process.
        """
        super().__init__()
        model_size = size.value if isinstance(size, self.Size) else size
        self.model_id = f"IDEA-Research/grounding-dino-{model_size}"
        self.device = device
        self.processor = AutoProcessor.from_pretrained(self.model_id)  # auto resizes to 800
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id, torch_dtype=precision_to_torch_dtype(precision)
            )
            .to(self.device)
            .eval()
        )
        self.model = optimize_model(
            model=self.model,
            device=self.device,
            precision=precision_to_torch_dtype(precision),
            compile_models=compile_models,
            verbose=verbose,
        )
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.template = template

    def __call__(
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

        # Run the dino model
        inputs = self.processor(images=pil_images, text=text_labels, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=self.model.dtype):
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
