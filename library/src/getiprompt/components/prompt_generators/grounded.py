"""Generate bounding boxes using a zero shot object detector."""

# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from difflib import SequenceMatcher
from enum import Enum

import torch
from torch import nn
from torchvision import tv_tensors
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from getiprompt.utils.utils import precision_to_torch_dtype


class GroundingModel(Enum):
    """The model to use for the grounding."""

    GROUNDING_DINO_BASE = "IDEA-Research/grounding-dino-base"
    GROUNDING_DINO_TINY = "IDEA-Research/grounding-dino-tiny"
    LLMDET_TINY = "fushh7/llmdet_swin_tiny_hf"
    LLMDET_BASE = "fushh7/llmdet_swin_base_hf"
    LLMDET_LARGE = "fushh7/llmdet_swin_large_hf"


class TextToBoxPromptGenerator(nn.Module):
    """Generates box prompts from text using a zero-shot object detector.

    All outputs are tensors for full traceability (ONNX/TorchScript compatible).

    Args:
        box_threshold: Confidence threshold for box detection.
        text_threshold: Confidence threshold for text matching.
        template: Template string for formatting category names.
        model_id: The grounding model to use.
        device: Device to run the model on.
        precision: Model precision (bf16, fp16, fp32).
        compile_models: Whether to compile the models.
        max_boxes: Maximum boxes per category for output padding. Default: 50.

    Examples:
        >>> import torch
        >>> from getiprompt.components.prompt_generators import TextToBoxPromptGenerator
        >>>
        >>> generator = TextToBoxPromptGenerator(
        ...     box_threshold=0.4,
        ...     text_threshold=0.3,
        ...     template=TextToBoxPromptGenerator.Template.specific_object,
        ... )
        >>> # category_mapping: {category_name: category_id}
        >>> category_mapping = {"cat": 1, "dog": 2}
        >>> box_prompts, num_boxes, category_ids = generator(images, category_mapping)
        >>> box_prompts.shape  # [T, C, max_boxes, 5]
        torch.Size([1, 2, 50, 5])
        >>> num_boxes.shape  # [T, C]
        torch.Size([1, 2])
        >>> category_ids.shape  # [C]
        torch.Size([2])
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
        model_id: GroundingModel = GroundingModel.LLMDET_TINY,
        device: str = "cuda",
        precision: str = "bf16",
        compile_models: bool = False,
        max_boxes: int = 50,
    ) -> None:
        """Initialize the TextToBoxPromptGenerator."""
        super().__init__()
        self.model_id = model_id.value
        self.device = device
        self.max_boxes = max_boxes
        self.model, self.processor = self._load_grounding_model_and_processor(
            self.model_id,
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
        precision: str,
        device: str,
        compile_models: bool,
    ) -> tuple[AutoModelForZeroShotObjectDetection, AutoProcessor]:
        """Load the grounding model and processor."""
        from getiprompt.utils.optimization import optimize_model

        processor = AutoProcessor.from_pretrained(model_id)
        if model_id.startswith("fushh7/llmdet_swin"):
            from getiprompt.models.foundation import GroundingDinoForObjectDetection

            model = GroundingDinoForObjectDetection.from_pretrained(
                model_id,
                torch_dtype=precision_to_torch_dtype(precision),
            )
        else:
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id,
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
        """Map labels to their best matching category by similarity."""
        processed_labels = []
        for label in labels:
            if label not in category_mapping:
                label = max(category_mapping.keys(), key=lambda x: SequenceMatcher(None, x, label).ratio())
            processed_labels.append(label)
        return processed_labels

    def _pad_boxes(self, boxes: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Pad boxes tensor to max_boxes size.

        Args:
            boxes: Boxes tensor [N, 5] with (x1, y1, x2, y2, score)
            device: Target device
            dtype: Target dtype

        Returns:
            Padded boxes tensor [max_boxes, 5]
        """
        num_boxes = boxes.shape[0] if boxes.numel() > 0 else 0
        if num_boxes >= self.max_boxes:
            return boxes[: self.max_boxes]

        padded = torch.zeros(self.max_boxes, 5, device=device, dtype=dtype)
        if num_boxes > 0:
            padded[:num_boxes] = boxes
        return padded

    def forward(
        self,
        target_images: list[tv_tensors.Image],
        category_mapping: dict[str, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate box prompts from text priors.

        Args:
            target_images: List of target images
            category_mapping: Mapping from category name to category ID

        Returns:
            box_prompts: [T, C, max_boxes, 5] - padded box prompts (x1, y1, x2, y2, score)
            category_ids: [C] - category ID mapping
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

        # Build category_ids tensor from mapping (sorted by category name for consistency)
        sorted_categories = sorted(category_mapping.keys())
        category_ids_list = [category_mapping[cat] for cat in sorted_categories]
        cat_name_to_idx = {cat: idx for idx, cat in enumerate(sorted_categories)}

        num_images = len(target_images)
        num_categories = len(category_ids_list)
        device = torch.device(self.device)
        dtype = torch.float32

        # Pre-allocate output tensors
        box_prompts = torch.zeros(num_images, num_categories, self.max_boxes, 5, device=device, dtype=dtype)
        category_ids = torch.tensor(category_ids_list, device=device, dtype=torch.int64)

        # Process each image's results
        for img_idx, result in enumerate(results):
            pred_labels = self._map_labels_to_categories(result["labels"], category_mapping)
            pred_bboxes = result["boxes"]
            pred_scores = result["scores"]

            # Group boxes by category
            boxes_per_category: dict[int, list[torch.Tensor]] = {idx: [] for idx in range(num_categories)}

            for pred_bbox, pred_score, pred_label in zip(pred_bboxes, pred_scores, pred_labels, strict=True):
                cat_idx = cat_name_to_idx[pred_label]
                box_with_score = torch.cat([pred_bbox, pred_score.unsqueeze(0)], dim=0)
                boxes_per_category[cat_idx].append(box_with_score)

            # Fill tensors for each category
            for cat_idx, boxes_list in boxes_per_category.items():
                if boxes_list:
                    boxes = torch.stack(boxes_list)
                    box_prompts[img_idx, cat_idx] = self._pad_boxes(boxes, device, dtype)

        return box_prompts, category_ids
