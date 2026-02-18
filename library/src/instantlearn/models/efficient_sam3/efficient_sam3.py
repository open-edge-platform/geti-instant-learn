# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientSAM3 model for text and visual prompting.

A student distilled variant of SAM3 using lightweight backbones (EfficientViT,
RepViT, TinyViT) and MobileCLIP text encoder for efficient inference.
"""

from __future__ import annotations

from itertools import zip_longest
from typing import TYPE_CHECKING

import torch
from transformers import CLIPTokenizerFast

from instantlearn.data.base.batch import Batch
from instantlearn.models.base import Model
from instantlearn.models.sam3.post_processing import PostProcessingConfig

if TYPE_CHECKING:
    from instantlearn.data.base.sample import Sample

# Reuse SAM3 processing pipeline (identical image size, normalization, postprocessing)
from instantlearn.models.sam3.processing import (
    Sam3Postprocessor as EfficientSam3Postprocessor,
)
from instantlearn.models.sam3.processing import (
    Sam3Preprocessor as EfficientSam3Preprocessor,
)
from instantlearn.models.sam3.processing import (
    Sam3PromptPreprocessor as EfficientSam3PromptPreprocessor,
)
from instantlearn.utils import precision_to_torch_dtype

from .constants import BACKBONE_CONFIG, STUDENT_CONTEXT_LENGTH
from .model import EfficientSam3Model


class EfficientSAM3(Model):
    """EfficientSAM3 model for text and visual prompting.

    Uses lightweight student backbones (EfficientViT, RepViT, TinyViT) distilled
    from SAM3, with MobileCLIP-S1 text encoder. Supports the same prompting
    patterns as SAM3 (text prompts and/or bounding boxes).

    Examples:
        >>> from instantlearn.models import EfficientSAM3
        >>> from instantlearn.data.base.sample import Sample
        >>> from instantlearn.data.base import Batch
        >>> import torch

        >>> model = EfficientSAM3(backbone_type="efficientvit", variant="b2")

        >>> # Text prompting
        >>> ref = Sample(categories=["cat", "dog"], category_ids=[0, 1])
        >>> model.fit(Batch.collate([ref]))
        >>> target = Batch.collate([Sample(image=torch.zeros(3, 640, 480))])
        >>> results = model.predict(target)
    """

    def __init__(
        self,
        backbone_type: str = "efficientvit",
        variant: str = "b2",
        device: str = "cuda",
        confidence_threshold: float = 0.4,
        resolution: int = 1008,
        precision: str = "fp32",
        post_processing: PostProcessingConfig | None = None,
    ) -> None:
        """Initialize the EfficientSAM3 model.

        Args:
            backbone_type: Vision backbone family. One of:
                'efficientvit' (variants: b0, b1, b2),
                'repvit' (variants: m0_9, m1_1, m2_3),
                'tinyvit' (variants: 5m, 11m, 21m).
            variant: Model size variant within the backbone family.
            device: Target device ('cuda', 'xpu', or 'cpu').
            confidence_threshold: Score threshold for filtering predictions.
                Default is 0.4, balancing precision and IoU across datasets.
                Sweep benchmarks (LVIS + PerSeg) showed ct=0.4 with IoM 0.3
                gives the best trade-off: strong mask quality (IoU) while
                keeping precision acceptable for real-time use.
            resolution: Input image resolution. Default: 1008.
            precision: Model precision ('fp32' or 'bf16').
            post_processing: Optional post-processing configuration for NMS,
                mask overlap removal, and non-overlapping pixel constraints.
                Default enables mask IoM suppression at 0.3 — this removes
                heavily overlapping low-quality masks without discarding
                legitimate multi-object detections. IoM outperformed NMS
                and other combinations in sweep benchmarks.

        Raises:
            ValueError: If backbone_type/variant is not supported.
        """
        super().__init__()

        key = (backbone_type, variant)
        if key not in BACKBONE_CONFIG:
            msg = f"Unsupported backbone: {backbone_type}/{variant}. Available: {list(BACKBONE_CONFIG.keys())}"
            raise ValueError(msg)

        # Default post-processing: mask IoM suppression at 0.3.
        # Sweep benchmarks (LVIS + PerSeg) showed IoM 0.3 removes
        # heavily-overlapping low-quality masks while preserving legitimate
        # multi-object detections. It outperformed NMS, non-overlap, and
        # all tested combinations on both precision and F1.
        if post_processing is None:
            post_processing = PostProcessingConfig(
                mask_iom_threshold=0.3,
                nms_iou_threshold=None,
                remove_overlapping_pixels=False,
            )

        self.backbone_type = backbone_type
        self.variant = variant
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution
        self.precision = precision

        self.category_mapping: dict[str, int] | None = None

        # Reuse SAM3 preprocessors (same image pipeline)
        self.image_preprocessor = EfficientSam3Preprocessor(target_size=resolution).to(device)
        self.prompt_preprocessor = EfficientSam3PromptPreprocessor(target_size=resolution).to(device)
        self.postprocessor = EfficientSam3Postprocessor(
            target_size=resolution,
            threshold=confidence_threshold,
            mask_threshold=0.5,
            post_processing=post_processing,
        ).to(device)

        # Reuse SAM3 CLIP tokenizer (same BPE vocabulary)
        # Use pad_token_id=0 to match the original SimpleTokenizer's zero-padding
        # behavior used during EfficientSAM3's distillation training.
        self.tokenizer = CLIPTokenizerFast.from_pretrained("jetjodh/sam3")
        self.tokenizer.pad_token_id = 0

        self.model = (
            EfficientSam3Model.from_pretrained(
                backbone_type=backbone_type,
                variant=variant,
                torch_dtype=precision_to_torch_dtype(precision),
            )
            .to(device)
            .eval()
        )

    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Store category mapping from reference data for text prompting.

        Args:
            reference: Reference data containing category names and IDs.
        """
        reference_batch = Batch.collate(reference)
        self.category_mapping = {}
        for sample in reference_batch.samples:
            for category_id, category in zip(
                sample.category_ids,
                sample.categories,
                strict=False,
            ):
                if category not in self.category_mapping:
                    self.category_mapping[category] = int(category_id)

    # TODO(refactor): _aggregate_results and predict() are near-identical to  # noqa: TD003, FIX002
    # SAM3. Extract shared logic into a common base class method during the
    # upcoming SAM3/EfficientSAM3 refactoring pass.

    @staticmethod
    def _aggregate_results(
        all_masks: list[torch.Tensor],
        all_boxes: list[torch.Tensor],
        all_labels: list[torch.Tensor],
        img_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """Aggregate results from multiple prompt predictions.

        Args:
            all_masks: List of mask tensors per prompt.
            all_boxes: List of box tensors per prompt.
            all_labels: List of label tensors per prompt.
            img_size: Original image size (height, width).

        Returns:
            Merged prediction dict with pred_masks, pred_boxes, pred_labels.
        """
        non_empty_masks = [m for m in all_masks if m.numel() > 0]
        non_empty_boxes = [b for b in all_boxes if b.numel() > 0]
        non_empty_labels = [label for label in all_labels if label.numel() > 0]

        if non_empty_masks:
            return {
                "pred_masks": torch.cat(non_empty_masks, dim=0),
                "pred_boxes": torch.cat(non_empty_boxes, dim=0),
                "pred_labels": torch.cat(non_empty_labels, dim=0),
            }

        return {
            "pred_masks": torch.empty(0, *img_size),
            "pred_boxes": torch.empty(0, 5),
            "pred_labels": torch.empty(0, dtype=torch.long),
        }

    def predict(
        self,
        target: Sample | list[Sample] | Batch,
    ) -> list[dict[str, torch.Tensor]]:
        """Perform inference on target images.

        If ``fit()`` was called, uses the stored category mapping. Otherwise,
        uses per-sample ``categories`` or ``bboxes``.

        Args:
            target: Target data to infer on.

        Returns:
            List of prediction dicts, one per image, with pred_masks, pred_boxes,
            pred_labels.
        """
        target_batch = Batch.collate(target)
        results = []
        use_fitted_categories = self.category_mapping is not None

        for sample in target_batch.samples:
            img_size = sample.image.shape[-2:]
            bboxes = sample.bboxes if sample.bboxes is not None else []

            image_tensor = sample.image.unsqueeze(0) if sample.image.ndim == 3 else sample.image
            with (
                torch.no_grad(),
                torch.autocast(device_type=self.device, dtype=precision_to_torch_dtype(self.precision)),
            ):
                pixel_values, original_sizes = self.image_preprocessor(
                    image_tensor.to(self.device),
                )
                vision_embeds = self.model.get_vision_features(pixel_values)

            if use_fitted_categories:
                texts = list(self.category_mapping.keys())
                category_ids = list(self.category_mapping.values())
            else:
                texts = sample.categories or []
                category_ids = sample.category_ids
                if len(bboxes) and len(texts) != len(bboxes):
                    texts = ["visual"] * len(bboxes)

            all_masks: list[torch.Tensor] = []
            all_boxes: list[torch.Tensor] = []
            all_labels: list[torch.Tensor] = []

            for text, bbox, cat_id in zip_longest(
                texts,
                bboxes,
                category_ids,
                fillvalue=None,
            ):
                text_inputs = self.tokenizer(
                    [text or "visual"],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=STUDENT_CONTEXT_LENGTH,
                    truncation=True,
                )
                input_ids = text_inputs.input_ids.to(self.device)
                attention_mask = text_inputs.attention_mask.to(self.device)

                input_boxes = None
                input_boxes_labels = None
                if bbox is not None:
                    input_boxes = self.prompt_preprocessor(bbox, original_sizes)
                    input_boxes_labels = torch.ones(
                        (1, 1),
                        dtype=torch.long,
                        device=self.device,
                    )

                with (
                    torch.no_grad(),
                    torch.autocast(device_type=self.device, dtype=precision_to_torch_dtype(self.precision)),
                ):
                    outputs = self.model(
                        vision_embeds=vision_embeds,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        input_boxes=input_boxes,
                        input_boxes_labels=input_boxes_labels,
                    )

                result = self.postprocessor(outputs, target_sizes=[img_size])
                boxes_with_scores = torch.cat(
                    [result[0]["boxes"], result[0]["scores"].unsqueeze(1)],
                    dim=1,
                )
                all_masks.append(result[0]["masks"].cpu())
                all_boxes.append(boxes_with_scores.cpu())
                all_labels.append(
                    torch.full(
                        (len(result[0]["boxes"]),),
                        cat_id,
                        dtype=torch.int64,
                    ),
                )

            results.append(
                self._aggregate_results(all_masks, all_boxes, all_labels, img_size),
            )

        return results

    @staticmethod
    def available_backbones() -> list[tuple[str, str]]:
        """List all supported backbone configurations.

        Returns:
            List of (backbone_type, variant) tuples.
        """
        return list(BACKBONE_CONFIG.keys())
