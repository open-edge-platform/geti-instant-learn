# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base classes for mask post-processing pipeline.

Post-processors are ``nn.Module`` subclasses that transform segmentation
predictions (masks, scores, labels) in a composable, chainable way.
All post-processors must use pure PyTorch operations should be ONNX/OpenVINO
exportable.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn

from instantlearn.components.sam.decoder import masks_to_boxes_traceable


class PostProcessor(nn.Module):
    """Abstract base for all mask post-processors.

    Every subclass must implement ``forward`` with the signature::

        forward(masks, scores, labels) -> (masks, scores, labels)

    Where:
        - masks:  ``[N, H, W]`` bool or float tensor
        - scores: ``[N]`` float tensor
        - labels: ``[N]`` int64 tensor

    All post-processors must use pure PyTorch operations should be ONNX/OpenVINO
    exportable.
    """

    @abstractmethod
    def forward(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply post-processing to segmentation predictions.

        Args:
            masks: Binary masks of shape ``[N, H, W]``.
            scores: Confidence scores of shape ``[N]``.
            labels: Category labels of shape ``[N]``.

        Returns:
            Tuple of (masks, scores, labels) with the same semantics,
            possibly fewer entries after filtering.
        """


class PostProcessorPipeline(PostProcessor):
    """Chains multiple :class:`PostProcessor` modules sequentially.

    Each processor is applied in order, with the output of one feeding
    into the next.

    Args:
        processors: Ordered list of post-processors to chain.

    Examples:
        >>> from instantlearn.components.postprocessing import (
        ...     MinimumAreaFilter,
        ...     MorphologicalOpening,
        ...     PostProcessorPipeline,
        ... )
        >>> pipeline = PostProcessorPipeline([
        ...     MinimumAreaFilter(min_area=64),
        ...     MorphologicalOpening(kernel_size=3),
        ... ])
    """

    def __init__(self, processors: list[PostProcessor]) -> None:
        """Initialize the pipeline with an ordered list of processors."""
        super().__init__()
        self.processors = nn.ModuleList(processors)

    def __len__(self) -> int:
        """Return the number of processors in the pipeline."""
        return len(self.processors)

    def forward(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply all processors in sequence.

        Args:
            masks: Binary masks ``[N, H, W]``.
            scores: Confidence scores ``[N]``.
            labels: Category labels ``[N]``.

        Returns:
            Post-processed (masks, scores, labels).
        """
        for processor in self.processors:
            masks, scores, labels = processor(masks, scores, labels)
        return masks, scores, labels


def apply_postprocessing(
    predictions: list[dict[str, torch.Tensor]],
    postprocessor: PostProcessor | None,
) -> list[dict[str, torch.Tensor]]:
    """Apply a post-processor to a list of prediction dicts.

    This helper unpacks the standard prediction dict, runs the
    post-processor, recomputes bounding boxes from cleaned masks,
    and repacks the result.

    Args:
        predictions: List of prediction dicts as returned by ``Model.predict()``.
        postprocessor: Post-processor to apply, or ``None`` to return as-is.

    Returns:
        Updated prediction dicts with post-processed masks and recomputed boxes.
    """
    if postprocessor is None:
        return predictions

    processed: list[dict[str, torch.Tensor]] = []
    for pred in predictions:
        orig_masks = pred["pred_masks"]
        scores = pred.get("pred_scores", torch.ones(orig_masks.size(0), device=orig_masks.device))
        labels = pred["pred_labels"]

        new_masks, new_scores, new_labels = postprocessor(orig_masks, scores, labels)

        result: dict[str, torch.Tensor] = {
            "pred_masks": new_masks,
            "pred_scores": new_scores,
            "pred_labels": new_labels,
        }

        # Only recompute boxes when masks were actually modified
        masks_changed = new_masks.shape != orig_masks.shape or not torch.equal(new_masks, orig_masks)

        if masks_changed:
            if new_masks.numel() > 0 and new_masks.size(0) > 0:
                boxes = masks_to_boxes_traceable(new_masks)
                box_scores = new_scores.unsqueeze(1)
                result["pred_boxes"] = torch.cat([boxes, box_scores], dim=1)
            else:
                result["pred_boxes"] = torch.empty(0, 5, device=new_masks.device)

        # Preserve any extra keys (e.g. pred_points, pred_boxes if unchanged)
        for key in pred:
            if key not in result:
                result[key] = pred[key]

        processed.append(result)

    return processed
