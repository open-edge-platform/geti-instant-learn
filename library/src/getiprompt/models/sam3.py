# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM3 model for text and visual prompting."""

from itertools import zip_longest

import torch
from transformers import CLIPTokenizerFast

from getiprompt.data.base.batch import Batch
from getiprompt.data.base.sample import Sample
from getiprompt.models.foundation.sam3 import Sam3ImageProcessorFast, Sam3Model, Sam3Processor
from getiprompt.utils.utils import precision_to_torch_dtype

from .base import Model


class SAM3(Model):
    """SAM3 model for text and visual prompting.

    This model uses SAM3 (Segment Anything Model 3) for zero-shot segmentation
    using either text prompts or visual prompts (bounding boxes).

    **Important: SAM3 differs from other prompt-based models** in that it does NOT
    require a separate learning phase. Instead, it performs zero-shot segmentation
    directly during inference using:
    - Text prompts (category names) provided via `fit()` or per-sample `categories`, OR
    - Visual prompts (bounding boxes) provided in the `bboxes` field of each sample

    At least one of these prompt types must be provided for each sample during inference.

    NOTE: Currently, SAM3 does not work well with torch.bfloat16 precision.

    Usage Patterns:
        **Pattern 1: Consistent text prompting via `fit()`**
        Use `fit()` to store categories, then `predict()` applies them to all images.

        **Pattern 2: Per-sample prompting**
        Skip `fit()` and provide categories/bboxes directly in each target sample.

    Examples:
        >>> from getiprompt.models import SAM3
        >>> from getiprompt.data.base import Batch
        >>> from getiprompt.data.base.sample import Sample
        >>> import torch
        >>> import numpy as np

        >>> sam3 = SAM3()

        >>> # Example 1: Using fit() to set category prompts directly in reference samples without passing reference images.
        >>> ref_sample = Sample(
        ...     categories=["shoe", "person"],
        ...     category_ids=[0, 1],
        ... )
        >>> sam3.fit(Batch.collate([ref_sample]))
        >>> target_batch = Batch.collate([Sample(image=torch.zeros((3, 1024, 1024)))])
        >>> infer_results = sam3.infer(target_batch)

        >>> # Example 2: Per-sample text prompting (without fit) but set category prompts in each target sample.
        >>> sam3_no_fit = SAM3()
        >>> target_sample = Sample(
        ...     image=torch.zeros((3, 1024, 1024)),
        ...     categories=["shoe", "person"],  # Category prompts per sample
        ...     category_ids=[0, 1],
        ... )
        >>> target_batch = Batch.collate([target_sample])
        >>> infer_results = sam3_no_fit.infer(target_batch)

        >>> # Example 3: Visual prompting with bounding boxes
        >>> target_sample = Sample(
        ...     image=torch.zeros((3, 1024, 1024)),
        ...     bboxes=np.array([[100, 100, 200, 200]]),  # [x, y, w, h]
        ... )
        >>> target_batch = Batch.collate([target_sample])
        >>> infer_results = sam3_no_fit.infer(target_batch)

        >>> isinstance(infer_results, list)
        True
    """

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        resolution: int = 1008,
        precision: str = "fp32",
        compile_models: bool = False,
    ) -> None:
        """Initialize the SAM3 model.

        Args:
            device: The device to use ('cuda', 'xpu', or 'cpu').
            confidence_threshold: The confidence threshold for filtering predictions.
            resolution: The input image resolution.
            precision: The precision to use for the model ('bf16' or 'fp32').
            compile_models: Whether to compile the models.
        """
        super().__init__()

        self.device = device
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution
        self.precision = precision
        self.compile_models = compile_models

        # Category mapping from fit() - optional for consistency with GroundedSAM
        self.category_mapping: dict[str, int] | None = None

        # Create processor manually with image processor and tokenizer
        image_processor = Sam3ImageProcessorFast.from_pretrained("facebook/sam3")
        tokenizer = CLIPTokenizerFast.from_pretrained("facebook/sam3")
        self.input_processor = Sam3Processor(
            image_processor=image_processor,
            tokenizer=tokenizer,
        )

        self.model = Sam3Model.from_pretrained(
            "facebook/sam3",
            key_mapping={r"detector_model.(.+)": r"\1"},
            torch_dtype=precision_to_torch_dtype(precision),
            attn_implementation="sdpa",
        ).to(device)

    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Store category mapping from reference batch for consistent API with GroundedSAM.

        This method is optional. If called, the stored categories will be used for all
        predictions. If not called, categories are taken from each target sample.

        Args:
            reference: Reference data to learn from. Accepts:
                - Sample: A single reference sample
                - list[Sample]: A list of reference samples
                - Batch: A batch of reference samples
        """
        reference_batch = Batch.collate(reference)
        self.category_mapping = {}
        for sample in reference_batch.samples:
            for category_id, category in zip(sample.category_ids, sample.categories, strict=False):
                if category not in self.category_mapping:
                    self.category_mapping[category] = int(category_id)

    def _aggregate_results(
        self,
        all_masks: list[torch.Tensor],
        all_boxes: list[torch.Tensor],
        all_labels: list[torch.Tensor],
        img_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """Aggregate results from multiple predictions.

        Args:
            all_masks: List of mask tensors.
            all_boxes: List of box tensors.
            all_labels: List of labels.
            img_size: The image size (height, width).

        Returns:
            Dictionary with aggregated predictions.
        """
        # Filter out empty tensors before concatenation
        non_empty_masks = [masks for masks in all_masks if masks.numel() > 0]
        non_empty_boxes = [boxes for boxes in all_boxes if boxes.numel() > 0]
        non_empty_labels = [labels for labels in all_labels if labels.numel() > 0]

        if non_empty_masks:
            aggregated_masks = torch.cat(non_empty_masks, dim=0)
            aggregated_boxes = torch.cat(non_empty_boxes, dim=0)
            aggregated_labels = torch.cat(non_empty_labels, dim=0)
        else:
            # No predictions found
            aggregated_masks = torch.empty(0, *img_size)
            aggregated_boxes = torch.empty(0, 5)
            aggregated_labels = torch.empty(0, dtype=torch.long)

        return {
            "pred_masks": aggregated_masks,
            "pred_boxes": aggregated_boxes,
            "pred_labels": aggregated_labels,
        }

    def predict(self, target: Sample | list[Sample] | Batch) -> list[dict[str, torch.Tensor]]:
        """Perform inference step on the target images.

        Uses batch image encoding for efficiency when processing multiple images.

        If `fit()` was called, uses the stored category mapping for text prompts.
        Otherwise, uses per-sample categories from target_batch.

        Args:
            target: Target data to infer. Accepts:
                - Sample: A single target sample
                - list[Sample]: A list of target samples
                - Batch: A batch of target samples
        """
        target_batch = Batch.collate(target)
        results = []
        samples = target_batch.samples

        # Use stored categories from fit() if available, otherwise use per-sample
        use_fitted_categories = self.category_mapping is not None

        # Process each image's prompts individually
        for sample in samples:
            img_size = sample.image.shape[-2:]
            bboxes = sample.bboxes if sample.bboxes is not None else []
            
            with torch.no_grad():
                img_inputs = self.input_processor(images=sample.image, return_tensors="pt").to(self.device)
                vision_embeds = self.model.get_vision_features(img_inputs.pixel_values)

            # Determine text prompts and category IDs
            if use_fitted_categories:
                texts = list(self.category_mapping.keys())
                category_ids = list(self.category_mapping.values())
            else:
                texts = sample.categories if sample.categories is not None else []
                category_ids = sample.category_ids

            all_masks: list[torch.Tensor] = []
            all_boxes: list[torch.Tensor] = []
            all_labels: list[torch.Tensor] = []
            for text, bbox, cat_id in zip_longest(texts, bboxes, category_ids, fillvalue=None):
                formatted_inputs = self.input_processor(
                    text=[text] if text is not None else None,
                    input_boxes=[bbox] if bbox is not None else None,
                    input_boxes_labels=len(bbox) * [1] if bbox is not None else None,
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model(
                        vision_embeds=vision_embeds,  # Reuse cached vision features
                        input_ids=formatted_inputs.input_ids if "input_ids" in formatted_inputs else None,
                        attention_mask=formatted_inputs.attention_mask if "attention_mask" in formatted_inputs else None,
                        input_boxes=formatted_inputs.input_boxes if "input_boxes" in formatted_inputs else None,
                        input_boxes_labels=formatted_inputs.input_boxes_labels if "input_boxes_labels" in formatted_inputs else None,
                    )
                result = self.input_processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.5,
                    mask_threshold=0.5,
                    target_sizes=img_inputs.get("original_sizes").tolist(),
                )
                boxes_with_scores = torch.cat(
                    [result[0]["boxes"], result[0]["scores"].unsqueeze(1)],
                    dim=1,
                )
                all_masks.append(result[0]["masks"].cpu())
                all_boxes.append(boxes_with_scores.cpu())
                all_labels.append(torch.full((len(result[0]["boxes"]),), cat_id, dtype=torch.int64))

            results.append(self._aggregate_results(all_masks, all_boxes, all_labels, img_size))

        return results
