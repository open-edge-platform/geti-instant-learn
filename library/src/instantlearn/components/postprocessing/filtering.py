# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Area-based mask filtering post-processor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from instantlearn.components.postprocessing.base import PostProcessor

if TYPE_CHECKING:
    import torch


class ScoreFilter(PostProcessor):
    """Remove masks with scores at or below a threshold.

    Designed to clean up "zombie" masks that the decoder kept at
    zero score (below ``confidence_threshold``) for ONNX-compatible
    fixed-shape tensors.  Place this as the **first step** in any
    pipeline that does not already discard zero-score masks (e.g.
    NMS-based pipelines).

    Args:
        min_score: Masks with scores ``<= min_score`` are removed.
            Default: ``0.0`` (removes only exact-zero scores).
    """

    def __init__(self, min_score: float = 0.0) -> None:
        """Initialize with the minimum score threshold."""
        super().__init__()
        self.min_score = min_score

    def forward(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter masks by minimum score.

        Args:
            masks: Binary masks ``[N, H, W]``.
            scores: Confidence scores ``[N]``.
            labels: Category labels ``[N]``.

        Returns:
            Filtered (masks, scores, labels).
        """
        if masks.size(0) == 0:
            return masks, scores, labels

        keep = scores > self.min_score
        return masks[keep], scores[keep], labels[keep]


class MinimumAreaFilter(PostProcessor):
    """Remove masks whose area is below a threshold.

    Straightforward filter that discards masks with fewer than
    ``min_area`` foreground pixels.

    Args:
        min_area: Minimum number of foreground pixels. Masks with
            area strictly less than this are removed. Default: ``100``.
    """

    def __init__(self, min_area: int = 100) -> None:
        """Initialize with the minimum mask area threshold."""
        super().__init__()
        self.min_area = min_area

    def forward(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter masks by minimum area.

        Args:
            masks: Binary masks ``[N, H, W]``.
            scores: Confidence scores ``[N]``.
            labels: Category labels ``[N]``.

        Returns:
            Filtered (masks, scores, labels).
        """
        if masks.size(0) == 0:
            return masks, scores, labels

        areas = masks.bool().flatten(1).sum(dim=1)  # [N]
        keep = areas >= self.min_area
        return masks[keep], scores[keep], labels[keep]
