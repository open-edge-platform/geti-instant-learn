"""Generate bounding boxes using a zero shot object detector."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from difflib import SequenceMatcher

import torch
from torch import nn
from torchvision import tv_tensors
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from getiprompt.utils.optimization import optimize_model
from getiprompt.utils.utils import precision_to_torch_dtype

AVAILABLE_GROUNDING_MODELS = {
    "grounding_dino_base": (
        "IDEA-Research/grounding-dino-base",
        "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
    ),
    "grounding_dino_tiny": (
        "IDEA-Research/grounding-dino-tiny",
        "a2bb814dd30d776dcf7e30523b00659f4f141c71",
    ),
    "llmdet_tiny": (
        "fushh7/llmdet_swin_tiny_hf",
        "6719e6ec2b2f7f0f3fed035fdca7b1856a1107c9",
    ),
    "llmdet_base": (
        "fushh7/llmdet_swin_base_hf",
        "a5dee636e9654c9c555b53092d8449cc82f48947",
    ),
    "llmdet_large": (
        "fushh7/llmdet_swin_large_hf",
        "d3523d16b1620552e5f386fbc851ea311e9b2bab",
    ),
}


class TextToBoxPromptGenerator(nn.Module):
    """This class generates text-to-box prompts for the segmenter."""

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
        model_id: str = "llmdet_tiny",
        device: str = "cuda",
        precision: str = "bf16",
        compile_models: bool = False,
    ) -> None:
        """Initialize the GroundingBoxGenerator.

        Get bounding box prompts from a grounding model.

        Args:
            box_threshold: The box threshold.
            text_threshold: The text threshold.
            template: The template to use for the prompt
            model_id: The grounding model to use.
            device: The device to use.
            precision: The precision to use for the model.
            compile_models: Whether to compile the models.

        Raises:
            ValueError: If the model ID is invalid.

        """
        super().__init__()

        if model_id not in AVAILABLE_GROUNDING_MODELS:
            valid_ids = list(AVAILABLE_GROUNDING_MODELS.keys())
            msg = f"Invalid model ID: {model_id}. Valid model IDs: {valid_ids}"
            raise ValueError(msg)

        hf_model_id, revision = AVAILABLE_GROUNDING_MODELS[model_id]
        self.model_id = hf_model_id
        self.device = device
        self.model, self.processor = self._load_grounding_model_and_processor(
            hf_model_id,
            revision,
            precision,
            device,
            compile_models,
        )
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.template = template

    @staticmethod
    def _load_grounding_model_and_processor(
        model_id: str,
        revision: str,
        precision: str,
        device: str,
        compile_models: bool,
    ) -> tuple[AutoModelForZeroShotObjectDetection, AutoProcessor]:
        """Load the grounding model and processor.

        Args:
            model_id: The model id to load.
            revision: Specific revision (commit SHA, tag, or branch) to pin
            precision: The precision to use for the model.
            device: The device to use for the model.
            compile_models: Whether to compile the models.
        """
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        if model_id.startswith("fushh7/llmdet_swin"):
            # LLMDET has a slightly different interface, use lazy import for efficiency
            from getiprompt.models.foundation import GroundingDinoForObjectDetection  # noqa: PLC0415

            model = GroundingDinoForObjectDetection.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=precision_to_torch_dtype(precision),
            )
        else:
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=precision_to_torch_dtype(precision),
            )
        model = optimize_model(
            model=model.to(device).eval(),
            device=device,
            precision=precision_to_torch_dtype(precision),
            compile_models=compile_models,
        )
        return model, processor

    @staticmethod
    def _map_labels_to_categories(labels: list[str], category_mapping: dict[str, int]) -> list[str]:
        """Map labels to their best matching category by similarity.

        Args:
            labels(list[str]): The labels to map to categories.
            category_mapping(dict[str, int]): The category mapping.

        Returns:
            list[str]: The mapped labels that match categories from ``category_mapping``.
        """
        processed_labels = []
        for label in labels:
            if label not in category_mapping:
                label_to_append = max(
                    category_mapping.keys(),
                    key=lambda x: SequenceMatcher(None, x, label).ratio(),
                )
            else:
                label_to_append = label
            processed_labels.append(label_to_append)
        return processed_labels

    def forward(
        self,
        target_images: list[tv_tensors.Image],
        category_mapping: dict[str, int],
    ) -> list[dict[int, torch.Tensor]]:
        """This generates bounding box prompt candidates based on the text priors.

        Args:
            target_images(list[tv_tensors.Image]): The target images
            category_mapping(dict[str, int]): The category mapping

        Returns:
            list[dict[int, torch.Tensor]]: List of prompts per image, one per target image instance.
        """
        formatted_categories = [self.template.format(prior=category) for category in category_mapping]
        prompts = ""
        for category in formatted_categories:
            prompts += category + ". "
        prompts = [prompts.strip()] * len(target_images)

        # Run the grounding model
        inputs = self.processor(images=target_images, text=prompts, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=self.model.dtype):
            outputs = self.model(**inputs)

        sizes = [image.shape[-2:] for image in target_images]
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=sizes,
        )

        # Generate all priors from the result of the Dino model.
        box_prompts: list[dict[int, torch.Tensor]] = []
        for result in results:
            pred_labels = self._map_labels_to_categories(result["labels"], category_mapping)
            pred_label_ids = [category_mapping[label] for label in pred_labels]
            pred_bboxes = result["boxes"]
            pred_scores = result["scores"]

            class_prompts: dict[int, torch.Tensor] = defaultdict(list)
            for pred_bbox, pred_score, pred_label_id in zip(
                pred_bboxes,
                pred_scores,
                pred_label_ids,
                strict=True,
            ):
                class_prompts[pred_label_id].append(torch.cat([pred_bbox, pred_score.unsqueeze(0)], dim=0))

            for class_id, prompts in class_prompts.items():
                class_prompts[class_id] = torch.stack(prompts)

            box_prompts.append(class_prompts)

        return box_prompts
