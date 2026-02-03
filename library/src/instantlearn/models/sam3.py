# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM3 model for text and visual prompting."""

from itertools import zip_longest

import torch
from torchvision.ops import box_convert

from instantlearn.data.base.batch import Batch
from instantlearn.data.base.sample import Sample
from instantlearn.models.foundation import Sam3Processor, build_sam3_image_model
from instantlearn.utils.utils import setup_autocast

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
        bpe_path: str | None = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        resolution: int = 1008,
        precision: str = "fp32",
        checkpoint_path: str | None = None,
        load_from_hf: bool = True,
        enable_segmentation: bool = True,
        enable_inst_interactivity: bool = False,
        compile_models: bool = False,
    ) -> None:
        """Initialize the SAM3 model.

        Args:
            bpe_path: Path to the BPE tokenizer vocabulary.
            device: The device to use ('cuda', 'xpu', or 'cpu').
            confidence_threshold: The confidence threshold for filtering predictions.
            resolution: The input image resolution.
            precision: The precision to use for the model ('bf16' or 'fp32').
            checkpoint_path: Optional path to model checkpoint.
            load_from_hf: Whether to load checkpoint from HuggingFace.
            enable_segmentation: Whether to enable segmentation head.
            enable_inst_interactivity: Whether to enable instance interactivity.
            compile_models: Whether to compile the models.
        """
        super().__init__()

        self.device = device
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution

        # Setup precision
        self.autocast_ctx = setup_autocast(device=device, precision=precision)

        # Build the SAM3 model
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=device,
            checkpoint_path=checkpoint_path,
            load_from_hf=load_from_hf,
            enable_segmentation=enable_segmentation,
            enable_inst_interactivity=enable_inst_interactivity,
            compile=compile_models,
        )

        # Create processor
        self.processor = Sam3Processor(
            model=self.model,
            resolution=resolution,
            device=device,
            confidence_threshold=confidence_threshold,
        )

        # Category mapping from fit() - optional for consistency with GroundedSAM
        self.category_mapping: dict[str, int] | None = None

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

    def _process_predictions(
        self,
        inference_state: dict,
        cat_id: int,
        img_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process predictions from inference state.

        Args:
            inference_state: The inference state containing predictions.
            cat_id: The category ID for the current prompt.
            img_size: The image size (height, width).

        Returns:
            Tuple of (processed_masks, boxes_with_scores, labels).
        """
        # Get predictions from state
        masks = inference_state.get("masks", torch.empty(0, *img_size))
        boxes = inference_state.get("boxes", torch.empty(0, 4))
        scores = inference_state.get("scores", torch.empty(0))
        labels = torch.full((boxes.shape[0],), cat_id, dtype=torch.long, device=boxes.device)
        # Process masks
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)  # [N, 1, H, W] -> [N, H, W]

        # Add scores to boxes
        if boxes.numel() > 0 and scores.numel() > 0:
            boxes = torch.cat([boxes, scores.unsqueeze(-1)], dim=-1)  # [N, 4] -> [N, 5]
        return masks, boxes, labels

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

    @staticmethod
    def normalize_boxes(boxes: torch.Tensor, img_size: tuple[int, int]) -> torch.Tensor:
        """Normalize boxes from absolute to relative coordinates."""
        img_h, img_w = img_size
        boxes = boxes.clone().to(torch.float32)
        boxes[:, [0, 2]] /= img_w  # x1, x2
        boxes[:, [1, 3]] /= img_h  # y1, y2
        return box_convert(boxes, "xyxy", "cxcywh")

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

        with self.autocast_ctx:
            # Batch encode all images at once (expensive backbone forward pass)
            images = [sample.image for sample in samples]
            batch_state = self.processor.set_image_batch(images)

            # Process each image's prompts individually
            for idx, sample in enumerate(samples):
                img_size = sample.image.shape[-2:]
                bboxes = self.normalize_boxes(sample.bboxes, img_size) if sample.bboxes is not None else []

                # Determine text prompts and category IDs
                if use_fitted_categories:
                    texts = list(self.category_mapping.keys())
                    category_ids = list(self.category_mapping.values())
                else:
                    texts = sample.categories if sample.categories is not None else []
                    category_ids = sample.category_ids

                # Extract single-image state from batch state
                inference_state = self.processor.get_single_image_state(batch_state, idx)

                all_masks: list[torch.Tensor] = []
                all_boxes: list[torch.Tensor] = []
                all_labels: list[torch.Tensor] = []
                for text, bbox, cat_id in zip_longest(texts, bboxes, category_ids, fillvalue=None):
                    inference_state = self.processor.set_prompt(inference_state, text=text, box=bbox)
                    pred_masks, pred_boxes, pred_labels = self._process_predictions(inference_state, cat_id, img_size)
                    all_masks.append(pred_masks)
                    all_boxes.append(pred_boxes)
                    all_labels.append(pred_labels)
                    self.processor.reset_all_prompts(inference_state)

                pred_result = self._aggregate_results(all_masks, all_boxes, all_labels, img_size)
                self.processor.reset_all_prompts(inference_state)
                results.append(pred_result)

        return results
