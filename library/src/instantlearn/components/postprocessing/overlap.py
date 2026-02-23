# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Overlap resolution post-processor using panoptic-style argmax assignment.

Resolves pixel-level overlaps between masks by assigning each pixel to the
highest-scoring mask. Masks that lose all pixels are removed.
"""

from __future__ import annotations

import torch

from instantlearn.components.postprocessing.base import PostProcessor


class PanopticArgmaxAssignment(PostProcessor):
    """Resolve overlapping masks via per-pixel argmax score assignment.

    For each pixel that belongs to more than one mask, only the
    highest-scored mask keeps that pixel. Masks that end up with
    fewer than ``min_area`` pixels are removed entirely.

    Adapted from SAM3's ``_apply_non_overlapping_constraints`` which
    uses the same argmax-based winner-takes-all approach.

    Args:
        min_area: Minimum area (in pixels) for a mask to survive
            after overlap resolution. Default: ``0`` (keep all).
    """

    def __init__(self, min_area: int = 0) -> None:
        """Initialize with optional minimum area after panoptic assignment."""
        super().__init__()
        self.min_area = min_area

    def forward(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign each pixel to its highest-scoring mask.

        Args:
            masks: Binary masks ``[N, H, W]``.
            scores: Confidence scores ``[N]``.
            labels: Category labels ``[N]``.

        Returns:
            Non-overlapping (masks, scores, labels).
        """
        if masks.size(0) <= 1:
            return masks, scores, labels

        n = masks.size(0)
        device = masks.device

        # Build score map: for each mask, fill its region with the mask's score.
        # Background (no mask) gets a large negative value.
        score_maps = masks.float() * scores[:, None, None]  # [N, H, W]

        # Where no mask covers a pixel, score_maps is 0 for all objects.
        # We use argmax — ties broken arbitrarily (first index).
        winner = torch.argmax(score_maps, dim=0)  # [H, W]

        # Build index map for broadcasting
        idx_map = torch.arange(n, device=device)[:, None, None]  # [N, 1, 1]

        # Each mask keeps only pixels where it is the winner AND was originally set
        resolved = (winner.unsqueeze(0) == idx_map) & masks.bool()  # [N, H, W]

        # Filter by minimum area
        areas = resolved.flatten(1).sum(dim=1)  # [N]
        keep = areas >= max(self.min_area, 1)  # at least 1 pixel

        return resolved[keep], scores[keep], labels[keep]
