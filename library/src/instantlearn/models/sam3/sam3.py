# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM3 model for text and visual prompting."""

from itertools import zip_longest

import torch
from transformers import CLIPTokenizerFast

from instantlearn.data.base.batch import Batch, Collatable
from instantlearn.data.base.sample import Sample
from instantlearn.models.base import Model
from instantlearn.utils import precision_to_torch_dtype

from .model import Sam3Model
from .processing import Sam3Postprocessor, Sam3Preprocessor, Sam3PromptPreprocessor


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
        >>> from instantlearn.models import SAM3
        >>> from instantlearn.data.base import Batch
        >>> from instantlearn.data.base.sample import Sample
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

        # Preprocessors and postprocessor
        self.image_preprocessor = Sam3Preprocessor(target_size=resolution).to(device)
        self.prompt_preprocessor = Sam3PromptPreprocessor(target_size=resolution).to(device)
        self.postprocessor = Sam3Postprocessor(
            target_size=resolution,
            threshold=confidence_threshold,
            mask_threshold=0.5,
        ).to(device)

        # Tokenizer for text prompts (still from transformers, but not used in ONNX path)
        self.tokenizer = CLIPTokenizerFast.from_pretrained("jetjodh/sam3")

        self.model = (
            Sam3Model.from_pretrained(
                "jetjodh/sam3",
                torch_dtype=precision_to_torch_dtype(precision),
            )
            .to(device)
            .eval()
        )

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

    @staticmethod
    def _aggregate_results(
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

    def predict(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Perform inference step on the target images.

        Uses batch image encoding for efficiency when processing multiple images.

        If `fit()` was called, uses the stored category mapping for text prompts.
        Otherwise, uses per-sample categories from target_batch.

        Args:
            target: Target data to infer. Accepts:
                - Sample: A single target sample
                - list[Sample]: A list of target samples
                - Batch: A batch of target samples
                - str | Path: A single image path
                - list[str] | list[Path]: Multiple image paths
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

            # Preprocess image
            image_tensor = sample.image.unsqueeze(0) if sample.image.ndim == 3 else sample.image
            with torch.no_grad():
                pixel_values, original_sizes = self.image_preprocessor(image_tensor.to(self.device))
                vision_embeds = self.model.get_vision_features(pixel_values)

            # Determine text prompts and category IDs
            if use_fitted_categories:
                texts = list(self.category_mapping.keys())
                category_ids = list(self.category_mapping.values())
            else:
                texts = sample.categories or []
                category_ids = sample.category_ids
                # Use "visual" placeholder when only bboxes are provided
                if len(bboxes) and len(texts) != len(bboxes):
                    texts = ["visual"] * len(bboxes)

            all_masks: list[torch.Tensor] = []
            all_boxes: list[torch.Tensor] = []
            all_labels: list[torch.Tensor] = []

            for text, bbox, cat_id in zip_longest(texts, bboxes, category_ids, fillvalue=None):
                # Tokenize text prompt (default to "visual" for bbox-only prompts)
                text_inputs = self.tokenizer([text or "visual"], return_tensors="pt", padding=True)
                input_ids = text_inputs.input_ids.to(self.device)
                attention_mask = text_inputs.attention_mask.to(self.device)

                # Prepare box inputs if bbox is provided (xyxy format)
                input_boxes = None
                input_boxes_labels = None
                if bbox is not None:
                    input_boxes = self.prompt_preprocessor(bbox, original_sizes)
                    input_boxes_labels = torch.ones((1, 1), dtype=torch.long, device=self.device)

                with torch.no_grad():
                    outputs = self.model(
                        vision_embeds=vision_embeds,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        input_boxes=input_boxes,
                        input_boxes_labels=input_boxes_labels,
                    )

                # Postprocess
                result = self.postprocessor(outputs, target_sizes=[img_size])
                boxes_with_scores = torch.cat(
                    [result[0]["boxes"], result[0]["scores"].unsqueeze(1)],
                    dim=1,
                )
                all_masks.append(result[0]["masks"].cpu())
                all_boxes.append(boxes_with_scores.cpu())
                all_labels.append(torch.full((len(result[0]["boxes"]),), cat_id, dtype=torch.int64))

            results.append(self._aggregate_results(all_masks, all_boxes, all_labels, img_size))

        return results
