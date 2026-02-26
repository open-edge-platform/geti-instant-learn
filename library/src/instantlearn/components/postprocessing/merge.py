# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Per-class and per-instance mask merging post-processors."""

from __future__ import annotations

import torch

from instantlearn.components.postprocessing.base import PostProcessor


def _union_find_root(parent: list[int], i: int) -> int:
    """Find root with path compression."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _union_find_merge(parent: list[int], rank: list[int], a: int, b: int) -> None:
    """Merge two sets by rank."""
    ra, rb = _union_find_root(parent, a), _union_find_root(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]:
        rank[ra] += 1


class InstanceMerge(PostProcessor):
    """Merge spatially connected same-label masks into single instances.

    Unlike :class:`MergePerClassMasks` which blindly merges *all* masks
    of the same class, this processor clusters masks by spatial proximity
    first, preserving separate instances of the same class.

    Two masks are considered connected if they share any foreground
    pixels (after optional dilation) **and** have the same label.
    Transitive connections are resolved via union-find: if A connects
    to B and B connects to C, then {A, B, C} form one instance.

    For each cluster the masks are OR-merged and the maximum score is
    kept.

    This is **not** ONNX-exportable because it uses dynamic clustering.

    Args:
        iou_threshold: Minimum mask-IoU between two masks to be
            considered spatially connected.  Default ``0.0`` means any
            pixel overlap is enough.
        gap_pixels: Dilate masks by this many pixels before checking
            overlap.  Bridges small spatial gaps (e.g. body-to-tail).
            Default: ``0`` (exact overlap only).

    Examples:
        >>> merger = InstanceMerge(gap_pixels=5)
        >>> # Two overlapping cat masks merge, a distant dog stays separate
        >>> merged_masks, merged_scores, merged_labels = merger(masks, scores, labels)
    """

    def __init__(self, iou_threshold: float = 0.0, gap_pixels: int = 0) -> None:
        """Initialize with overlap threshold and optional dilation gap."""
        super().__init__()
        self.iou_threshold = iou_threshold
        self.gap_pixels = gap_pixels

    @property
    def exportable(self) -> bool:
        """Not exportable due to dynamic clustering."""
        return False

    def forward(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge spatially connected same-label masks.

        Args:
            masks: Binary masks ``[N, H, W]``.
            scores: Confidence scores ``[N]``.
            labels: Category labels ``[N]``.

        Returns:
            Merged (masks, scores, labels) — one entry per instance.
        """
        n = masks.size(0)
        if n == 0:
            return masks, scores, labels

        # Optionally dilate masks to bridge small gaps
        if self.gap_pixels > 0:
            ks = 2 * self.gap_pixels + 1
            kernel = torch.ones(1, 1, ks, ks, device=masks.device)
            dilated = (
                torch.nn.functional.conv2d(
                    masks.float().unsqueeze(1),
                    kernel,
                    padding=self.gap_pixels,
                )
                > 0
            ).squeeze(1)
        else:
            dilated = masks.bool()

        # Build union-find by pairwise overlap within same label
        parent = list(range(n))
        rank = [0] * n

        flat = dilated.flatten(1).float()  # [N, H*W]

        for i in range(n):
            for j in range(i + 1, n):
                # Skip different labels
                if labels[i] != labels[j]:
                    continue

                # Compute overlap
                intersection = (flat[i] * flat[j]).sum()
                if self.iou_threshold > 0:
                    union = flat[i].sum() + flat[j].sum() - intersection
                    iou = intersection / (union + 1e-6)
                    if iou > self.iou_threshold:
                        _union_find_merge(parent, rank, i, j)
                elif intersection > 0:
                    _union_find_merge(parent, rank, i, j)

        # Collect clusters
        clusters: dict[int, list[int]] = {}
        for i in range(n):
            root = _union_find_root(parent, i)
            clusters.setdefault(root, []).append(i)

        # Merge each cluster
        merged_masks = []
        merged_scores = []
        merged_labels = []

        for indices in clusters.values():
            idx = torch.tensor(indices, device=masks.device)
            cluster_masks = masks[idx]
            cluster_scores = scores[idx]

            # OR-merge
            merged = (cluster_masks.float().sum(0) > 0).unsqueeze(0)
            max_score = cluster_scores.max().unsqueeze(0)

            merged_masks.append(merged)
            merged_scores.append(max_score)
            merged_labels.append(labels[indices[0]].unsqueeze(0))

        return (
            torch.cat(merged_masks),
            torch.cat(merged_scores),
            torch.cat(merged_labels),
        )


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
