# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base classes for mask post-processing pipeline.

Post-processors are ``nn.Module`` subclasses that transform segmentation
predictions (masks, scores, labels) in a composable, chainable way.
Each post-processor declares whether it is safe for ONNX export via
the ``exportable`` property, allowing the pipeline to automatically
exclude non-traceable operations at export time.
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

    The ``exportable`` property should be overridden to ``False`` for
    processors that rely on non-traceable ops (e.g. OpenCV, SciPy).
    """

    @property
    def exportable(self) -> bool:
        """Whether this processor uses only ONNX-traceable operations."""
        return True

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

    @property
    def exportable(self) -> bool:
        """True only if every child processor is exportable."""
        return all(p.exportable for p in self.processors)

    def exportable_subset(self) -> PostProcessorPipeline:
        """Return a new pipeline containing only ONNX-exportable processors.

        Useful at export time to include only traceable post-processing
        in the exported graph while keeping non-traceable ops for eager mode.

        Returns:
            A new ``PostProcessorPipeline`` with non-exportable processors removed.
        """
        return PostProcessorPipeline([p for p in self.processors if p.exportable])

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
        masks = pred["pred_masks"]
        scores = pred.get("pred_scores", torch.ones(masks.size(0), device=masks.device))
        labels = pred["pred_labels"]

        masks, scores, labels = postprocessor(masks, scores, labels)

        result: dict[str, torch.Tensor] = {
            "pred_masks": masks,
            "pred_scores": scores,
            "pred_labels": labels,
        }

        # Recompute boxes from cleaned masks
        if masks.numel() > 0 and masks.size(0) > 0:
            boxes = masks_to_boxes_traceable(masks)
            box_scores = scores.unsqueeze(1)
            result["pred_boxes"] = torch.cat([boxes, box_scores], dim=1)
        else:
            result["pred_boxes"] = torch.empty(0, 5, device=masks.device)

        # Preserve any extra keys (e.g. pred_points)
        for key in pred:
            if key not in result:
                result[key] = pred[key]

        processed.append(result)

    return processed
