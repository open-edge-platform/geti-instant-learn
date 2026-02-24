# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Per-class mask merging post-processor."""

from __future__ import annotations

import torch

from instantlearn.components.postprocessing.base import PostProcessor


class MergePerClassMasks(PostProcessor):
    """Merge all masks sharing the same label into a single mask per class.

    For each unique label, all corresponding masks are OR-merged into one
    binary mask and the maximum score among them is kept. This replicates
    the old ``SamDecoder.merge_masks_per_class=True`` behavior but as a
    composable pipeline step — place it *after* NMS, morphology, etc.

    The merger discards zero-score entries (score ``<= 0``), which matches
    SamDecoder's convention of zeroing out filtered masks.

    Examples:
        >>> from instantlearn.components.postprocessing import MergePerClassMasks
        >>> merger = MergePerClassMasks()
        >>> masks = torch.ones(3, 64, 64, dtype=torch.bool)
        >>> scores = torch.tensor([0.8, 0.6, 0.9])
        >>> labels = torch.tensor([0, 0, 1])
        >>> merged_masks, merged_scores, merged_labels = merger(masks, scores, labels)
        >>> merged_masks.shape[0]  # 2 unique classes
        2
    """

    def forward(  # noqa: PLR6301
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge masks per class.

        Args:
            masks: Binary masks ``[N, H, W]``.
            scores: Confidence scores ``[N]``.
            labels: Category labels ``[N]``.

        Returns:
            Merged (masks, scores, labels) with one entry per unique label.
        """
        if masks.size(0) == 0:
            return masks, scores, labels

        # Filter out invalid entries (score <= 0)
        valid = scores > 0
        masks = masks[valid]
        scores = scores[valid]
        labels = labels[valid]

        if masks.size(0) == 0:
            return (
                torch.empty(0, *masks.shape[1:], device=masks.device, dtype=masks.dtype),
                torch.empty(0, device=scores.device, dtype=scores.dtype),
                torch.empty(0, device=labels.device, dtype=labels.dtype),
            )

        unique_labels = torch.unique(labels)
        merged_masks = []
        merged_scores = []
        merged_labels = []

        for label in unique_labels:
            mask_for_label = labels == label
            class_masks = masks[mask_for_label]
            class_scores = scores[mask_for_label]

            # OR-merge all masks for this class
            merged = (class_masks.float().sum(0) > 0).unsqueeze(0)
            max_score = class_scores.max().unsqueeze(0)

            merged_masks.append(merged)
            merged_scores.append(max_score)
            merged_labels.append(label.unsqueeze(0))

        return (
            torch.cat(merged_masks),
            torch.cat(merged_scores),
            torch.cat(merged_labels),
        )
