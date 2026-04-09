# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configurable post-processing for SAM3 and EfficientSAM3 predictions.

Provides three independently toggleable algorithms to suppress duplicate and
overlapping predictions:

- **Box NMS**: Standard non-maximum suppression on bounding boxes via
  ``torchvision.ops.nms``. Fast and effective for box-level deduplication.
- **Mask IoM suppression**: Greedy overlap removal using Intersection-over-
  Minimum-area (IoM) on binary masks. Ported from the upstream SAM3
  ``remove_overlapping_masks`` implementation. Better than IoU for nested masks
  (e.g. a small object inside a larger one).
- **Non-overlapping pixel constraint**: Pixel-level winner-take-all — at each
  pixel, only the highest-scored mask retains its value. Ported from the
  upstream SAM3 ``_apply_non_overlapping_constraints``.

All algorithms share the same ``(scores, boxes, masks) → (scores, boxes, masks)``
interface, orchestrated by :func:`apply_post_processing`.

Example:
    >>> from instantlearn.models.sam3.post_processing import PostProcessingConfig
    >>> from instantlearn.models import EfficientSAM3
    >>> config = PostProcessingConfig(nms_iou_threshold=0.5, mask_iom_threshold=0.3)
    >>> model = EfficientSAM3(post_processing=config)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torchvision.ops import nms

logger = logging.getLogger(__name__)


@dataclass
class PostProcessingConfig:
    """Configuration for post-processing of SAM3/EfficientSAM3 predictions.

    Each field controls an independent post-processing algorithm. Set a
    threshold to enable that algorithm; leave as ``None`` (or ``False``) to
    disable it. Multiple algorithms can be combined — they are applied in
    order: box NMS → mask IoM suppression → non-overlapping pixel constraint.

    Args:
        nms_iou_threshold: IoU threshold for box-level NMS via
            ``torchvision.ops.nms``. Predictions with IoU > threshold against
            a higher-scored prediction are suppressed. Typical values:
            0.5-0.7. ``None`` disables box NMS.
        mask_iom_threshold: IoM (Intersection-over-Minimum-area) threshold
            for greedy mask overlap removal. A mask is discarded if its IoM
            with any higher-scored kept mask exceeds this threshold. Typical
            value: 0.3 (upstream SAM3 default). ``None`` disables mask IoM
            suppression.
        non_overlapping_masks: When ``True``, applies pixel-level argmax
            constraint: at each pixel, only the highest-scored mask retains
            its value; all others are zeroed. Useful for instance segmentation
            where masks should not overlap.

    Examples:
        Enable only box NMS:

        >>> config = PostProcessingConfig(nms_iou_threshold=0.5)

        Enable mask IoM suppression (upstream SAM3 default):

        >>> config = PostProcessingConfig(mask_iom_threshold=0.3)

        Enable all three algorithms:

        >>> config = PostProcessingConfig(
        ...     nms_iou_threshold=0.5,
        ...     mask_iom_threshold=0.3,
        ...     non_overlapping_masks=True,
        ... )
    """

    nms_iou_threshold: float | None = None
    mask_iom_threshold: float | None = None
    non_overlapping_masks: bool = False


def box_nms(
    scores: torch.Tensor,
    boxes: torch.Tensor,
    masks: torch.Tensor,
    iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply non-maximum suppression on bounding boxes.

    Wraps ``torchvision.ops.nms`` and filters scores, boxes, and masks to
    keep only the surviving predictions, ordered by descending score.

    Args:
        scores: Confidence scores ``(N,)``.
        boxes: Bounding boxes ``(N, 4)`` in ``(x1, y1, x2, y2)`` format.
            If boxes have a 5th column (score), only the first 4 are used.
        masks: Binary masks ``(N, H, W)``.
        iou_threshold: IoU threshold for suppression.

    Returns:
        Filtered ``(scores, boxes, masks)`` tuple with suppressed predictions
        removed, ordered by descending score.
    """
    if scores.numel() == 0:
        return scores, boxes, masks

    # torchvision.ops.nms expects (N, 4) boxes in xyxy format
    nms_boxes = boxes[:, :4].float()
    keep = nms(nms_boxes, scores.float(), iou_threshold)

    return scores[keep], boxes[keep], masks[keep]


def _mask_intersection(
    masks1: torch.Tensor,
    masks2: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    """Compute pairwise pixel intersection between two sets of binary masks.

    Processes in blocks to avoid OOM on large mask sets. Ported from the
    upstream SAM3 ``mask_intersection`` function.

    Args:
        masks1: Binary masks ``(N, H, W)`` (bool or int).
        masks2: Binary masks ``(M, H, W)`` (bool or int).
        block_size: Block size for chunked computation.

    Returns:
        Intersection counts ``(N, M)`` as long tensor.
    """
    masks1 = masks1.bool()
    masks2 = masks2.bool()
    n, m = masks1.shape[0], masks2.shape[0]
    out = torch.zeros(n, m, device=masks1.device, dtype=torch.long)

    for i in range(0, n, block_size):
        for j in range(0, m, block_size):
            a = masks1[i : i + block_size]  # (block, H, W)
            b = masks2[j : j + block_size]  # (block, H, W)
            inter = (a[:, None] & b[None, :]).flatten(-2).sum(-1)
            out[i : i + block_size, j : j + block_size] = inter

    return out


def _mask_iom(
    masks1: torch.Tensor,
    masks2: torch.Tensor,
) -> torch.Tensor:
    """Compute Intersection-over-Minimum-area (IoM) between two mask sets.

    IoM = intersection / min(area1, area2). Unlike IoU, IoM is sensitive to
    containment — a small mask fully inside a larger one gives IoM = 1.0
    regardless of the larger mask's area.

    Args:
        masks1: Binary masks ``(N, H, W)``.
        masks2: Binary masks ``(M, H, W)``.

    Returns:
        IoM matrix ``(N, M)`` as float tensor.
    """
    inter = _mask_intersection(masks1, masks2)  # (N, M)
    area1 = masks1.bool().flatten(-2).sum(-1)  # (N,)
    area2 = masks2.bool().flatten(-2).sum(-1)  # (M,)
    min_area = torch.min(area1[:, None], area2[None, :]).clamp_min(1)
    return inter.float() / min_area.float()


def mask_iom_suppression(
    scores: torch.Tensor,
    boxes: torch.Tensor,
    masks: torch.Tensor,
    iom_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy mask overlap removal using IoM (Intersection-over-Minimum-area).

    Sorts predictions by score descending. For each candidate, computes its
    IoM against all already-kept masks. If IoM exceeds the threshold against
    any kept mask, the candidate is suppressed. This is ported from the
    upstream SAM3 ``remove_overlapping_masks`` function.

    Args:
        scores: Confidence scores ``(N,)``.
        boxes: Bounding boxes ``(N, 4)`` or ``(N, 5)``.
        masks: Binary masks ``(N, H, W)``.
        iom_threshold: Maximum allowed IoM with any kept mask.

    Returns:
        Filtered ``(scores, boxes, masks)`` with overlapping duplicates removed,
        ordered by descending score.
    """
    n = scores.shape[0]
    if n <= 1:
        return scores, boxes, masks

    # Sort by score descending
    order = scores.argsort(descending=True)
    sorted_masks = masks[order]

    kept_indices: list[int] = []
    kept_masks_list: list[torch.Tensor] = []

    for i in range(n):
        candidate = sorted_masks[i].unsqueeze(0)  # (1, H, W)

        if not kept_masks_list:
            kept_indices.append(i)
            kept_masks_list.append(sorted_masks[i])
            continue

        kept_stack = torch.stack(kept_masks_list, dim=0)  # (K, H, W)
        iom_vals = _mask_iom(candidate, kept_stack).squeeze(0)  # (K,)

        if not torch.any(iom_vals > iom_threshold):
            kept_indices.append(i)
            kept_masks_list.append(sorted_masks[i])

    keep = order[kept_indices]
    return scores[keep], boxes[keep], masks[keep]


def apply_non_overlapping_constraint(
    scores: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Apply pixel-level non-overlapping constraint across masks.

    At each pixel, only the mask with the highest score retains its value;
    all other masks are zeroed at that pixel. Ported from the upstream SAM3
    ``_apply_non_overlapping_constraints``.

    This is useful for instance segmentation where each pixel should belong
    to at most one predicted instance.

    Args:
        scores: Confidence scores ``(N,)`` used to determine priority.
        masks: Binary masks ``(N, H, W)`` (int or bool).

    Returns:
        Modified masks ``(N, H, W)`` where overlapping pixels are assigned
        to the highest-scored mask only.
    """
    n = masks.shape[0]
    if n <= 1:
        return masks

    # Create score-weighted masks for argmax: each mask pixel gets the
    # object's score, background pixels get -inf
    score_map = scores[:, None, None].float() * masks.float()  # (N, H, W)
    # Set non-mask pixels to -inf so they never win argmax
    score_map[masks == 0] = float("-inf")

    # At each pixel, find which mask has the highest score
    winner = score_map.argmax(dim=0)  # (H, W), values in [0, N-1]

    # Build output: only the winner keeps its mask value at each pixel
    idx = torch.arange(n, device=masks.device)[:, None, None]  # (N, 1, 1)
    result = masks.clone()
    result[idx != winner.unsqueeze(0)] = 0

    return result


def apply_post_processing(
    scores: torch.Tensor,
    boxes: torch.Tensor,
    masks: torch.Tensor,
    config: PostProcessingConfig | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply configured post-processing algorithms in sequence.

    Algorithms are applied in order:
    1. Box NMS (if ``config.nms_iou_threshold`` is set)
    2. Mask IoM suppression (if ``config.mask_iom_threshold`` is set)
    3. Non-overlapping pixel constraint (if ``config.non_overlapping_masks``)

    Args:
        scores: Confidence scores ``(N,)``.
        boxes: Bounding boxes ``(N, 4)`` or ``(N, 5)``.
        masks: Binary masks ``(N, H, W)``.
        config: Post-processing configuration. If ``None``, inputs are
            returned unchanged.

    Returns:
        Post-processed ``(scores, boxes, masks)`` tuple.
    """
    if config is None or scores.numel() == 0:
        return scores, boxes, masks

    if config.nms_iou_threshold is not None:
        scores, boxes, masks = box_nms(scores, boxes, masks, config.nms_iou_threshold)

    if config.mask_iom_threshold is not None:
        scores, boxes, masks = mask_iom_suppression(scores, boxes, masks, config.mask_iom_threshold)

    if config.non_overlapping_masks:
        masks = apply_non_overlapping_constraint(scores, masks)

    return scores, boxes, masks
