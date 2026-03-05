# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Non-Maximum Suppression post-processors for segmentation masks.

Provides NMS variants operating on mask IoU, box IoU, mask IoM
(Intersection over Minimum), box IoM, and Soft-NMS with Gaussian score decay.
All implementations use pure PyTorch operations and are ONNX-exportable,
except :class:`BoxNMS` which depends on ``torchvision.ops.nms``.
"""

from __future__ import annotations

import torch
from torchvision.ops import nms as torchvision_nms

from instantlearn.components.postprocessing.base import PostProcessor
from instantlearn.components.sam.decoder import masks_to_boxes_traceable


def _pairwise_mask_iou(masks: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between all mask pairs.

    Args:
        masks: Boolean masks ``[N, H, W]``.

    Returns:
        IoU matrix ``[N, N]`` with values in ``[0, 1]``.
    """
    masks_flat = masks.flatten(1).float()  # [N, H*W]
    intersection = masks_flat @ masks_flat.T  # [N, N]
    areas = masks_flat.sum(dim=1)  # [N]
    union = areas.unsqueeze(0) + areas.unsqueeze(1) - intersection
    return intersection / (union + 1e-6)


def _pairwise_mask_iom(masks: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoM (Intersection over Minimum area) between all mask pairs.

    IoM is defined as ``intersection / min(area_i, area_j)``. This metric
    is better than IoU for nested objects (e.g. a cup on a table) because
    a small mask fully contained in a large one gets IoM=1 but IoU<<1.

    Args:
        masks: Boolean masks ``[N, H, W]``.

    Returns:
        IoM matrix ``[N, N]`` with values in ``[0, 1]``.
    """
    masks_flat = masks.flatten(1).float()  # [N, H*W]
    intersection = masks_flat @ masks_flat.T  # [N, N]
    areas = masks_flat.sum(dim=1)  # [N]
    min_areas = torch.minimum(areas.unsqueeze(0), areas.unsqueeze(1))
    return intersection / (min_areas + 1e-6)


def _pairwise_box_iom(boxes: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoM (Intersection over Minimum area) between boxes.

    IoM is defined as ``intersection_area / min(area_i, area_j)``.
    Boxes are in ``(x1, y1, x2, y2)`` format.

    Args:
        boxes: Bounding boxes ``[N, 4]``.

    Returns:
        IoM matrix ``[N, N]`` with values in ``[0, 1]``.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)  # [N]

    # Pairwise intersection
    inter_x1 = torch.maximum(x1.unsqueeze(0), x1.unsqueeze(1))  # [N, N]
    inter_y1 = torch.maximum(y1.unsqueeze(0), y1.unsqueeze(1))
    inter_x2 = torch.minimum(x2.unsqueeze(0), x2.unsqueeze(1))
    inter_y2 = torch.minimum(y2.unsqueeze(0), y2.unsqueeze(1))
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    min_areas = torch.minimum(areas.unsqueeze(0), areas.unsqueeze(1))
    return inter_area / (min_areas + 1e-6)


def _greedy_nms(
    scores: torch.Tensor,
    overlap_matrix: torch.Tensor,
    threshold: float,
    areas: torch.Tensor | None = None,
    score_margin: float | None = None,
    area_ratio: float = 0.5,
) -> torch.Tensor:
    """Greedy NMS using a precomputed overlap matrix.

    Supports **containment-aware suppression** when both ``areas`` and
    ``score_margin`` are provided.  In standard greedy NMS, a small
    high-scored mask (e.g. a cat's eye) suppresses a large lower-scored
    mask (e.g. the whole cat body) because the eye overlaps heavily
    with the body (IoM ≈ 1).  With containment-awareness enabled, if
    the winner is *truly contained inside* the loser (winner area <
    loser area * ``area_ratio``) *and* the score difference is within
    ``score_margin``, the larger mask is preferred and the smaller one
    is suppressed instead.

    The ``area_ratio`` guard prevents false containment swaps between
    similarly-sized masks (e.g. two body masks at 90 k vs 92 k pixels).
    Only when the winner is **significantly** smaller than the loser
    (e.g. eye 2 k inside body 90 k → ratio 0.02 < 0.5) does the swap
    trigger.

    Args:
        scores: Confidence scores ``[N]``.
        overlap_matrix: Pairwise overlap values ``[N, N]`` (IoU or IoM).
        threshold: Overlap threshold for suppression.
        areas: Per-mask areas ``[N]``.  Required for containment check.
        score_margin: Maximum score gap below which the larger mask is
            preferred over the smaller contained one.  ``None`` disables
            the containment logic (standard greedy NMS).
        area_ratio: Maximum winner/loser area ratio that counts as true
            containment.  The swap only triggers when
            ``areas[winner] < areas[loser] * area_ratio``.
            Default: ``0.5`` (winner must be less than half the loser's
            area).

    Returns:
        Indices of kept masks as a 1-D int64 tensor.

    Note:
        When containment-aware mode swaps out a winner ``idx`` (marking it
        suppressed), any masks that ``idx`` already suppressed in the current
        inner-loop iteration remain suppressed. This is intentional: those
        masks had lower scores than ``idx`` and would typically also be
        suppressed by the replacement winner ``jdx`` (which has a similar
        score). Re-evaluating them would add complexity with negligible
        practical benefit.
    """
    order = torch.argsort(scores, descending=True)
    kept: list[int] = []
    n = scores.size(0)

    suppressed = torch.zeros(n, dtype=torch.bool, device=scores.device)

    containment_aware = areas is not None and score_margin is not None

    for i in range(n):
        idx = order[i].item()
        if suppressed[idx]:
            continue
        kept.append(idx)

        # Suppress all lower-scored masks that overlap too much
        for j in range(i + 1, n):
            jdx = order[j].item()
            if suppressed[jdx]:
                continue
            if overlap_matrix[idx, jdx] <= threshold:
                continue

            # Overlap exceeds threshold — decide who gets suppressed
            if (
                containment_aware
                and areas[idx] < areas[jdx] * area_ratio
                and (scores[idx] - scores[jdx]) < score_margin
            ):
                # Winner (idx) is truly contained inside loser (jdx):
                # its area is less than area_ratio * the loser's area.
                # Score gap is within margin → prefer the larger mask.
                suppressed[idx] = True
                # Remove idx from kept and replace with jdx
                kept.remove(idx)
                # Don't suppress jdx — it will be picked up in a later
                # iteration (or this one re-processes it).
                break
            suppressed[jdx] = True

    return torch.tensor(kept, dtype=torch.int64, device=scores.device)


class MaskNMS(PostProcessor):
    """Non-Maximum Suppression using pairwise mask IoU.

    Greedy suppression: iterates masks by descending score, discards
    any mask whose IoU with an already-kept mask exceeds ``iou_threshold``
    as done in SAM3 paper.

    Operates directly on binary masks rather than bounding boxes, so it
    handles non-rectangular shapes correctly.

    Args:
        iou_threshold: IoU threshold above which a lower-scored mask
            is suppressed. Default: ``0.5``.
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        """Initialize MaskNMS with the given IoU threshold."""
        super().__init__()
        self.iou_threshold = iou_threshold

    def forward(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply mask-IoU NMS.

        Args:
            masks: Binary masks ``[N, H, W]``.
            scores: Confidence scores ``[N]``.
            labels: Category labels ``[N]``.

        Returns:
            Filtered (masks, scores, labels).
        """
        if masks.size(0) <= 1:
            return masks, scores, labels

        iou_matrix = _pairwise_mask_iou(masks.bool())
        keep = _greedy_nms(scores, iou_matrix, self.iou_threshold)
        return masks[keep], scores[keep], labels[keep]


class BoxNMS(PostProcessor):
    """Non-Maximum Suppression using bounding-box IoU via torchvision.

    Boxes are derived from masks using :func:`masks_to_boxes_traceable`.
    This is the fastest NMS variant but may be inaccurate for
    non-rectangular masks.

    .. note::
        This processor depends on ``torchvision.ops.nms`` and may not be
        ONNX-exportable. For an ONNX-safe alternative, use :class:`BoxIoMNMS`.

    Args:
        iou_threshold: IoU threshold for box overlap. Default: ``0.5``.
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        """Initialize BoxNMS with the given IoU threshold."""
        super().__init__()
        self.iou_threshold = iou_threshold

    def forward(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply box NMS.

        Args:
            masks: Binary masks ``[N, H, W]``.
            scores: Confidence scores ``[N]``.
            labels: Category labels ``[N]``.

        Returns:
            Filtered (masks, scores, labels).
        """
        if masks.size(0) <= 1:
            return masks, scores, labels

        boxes = masks_to_boxes_traceable(masks)
        keep = torchvision_nms(boxes.float(), scores.float(), self.iou_threshold)
        return masks[keep], scores[keep], labels[keep]


class BoxIoMNMS(PostProcessor):
    """Non-Maximum Suppression using bounding-box IoM (Intersection over Minimum).

    Similar to :class:`MaskIoMNMS` but operates on bounding boxes derived from masks,
    making it faster while still handling nested/contained objects well.
    A small box fully inside a larger one gets IoM ≈ 1.0 (suppressed),
    whereas IoU would be low (kept).

    Supports **containment-aware** mode via ``score_margin``.  When the
    higher-scored mask is *contained inside* a lower-scored mask and the
    score gap is within the margin, the larger mask is preferred.  This
    prevents tiny high-confidence fragments (e.g. an eye) from
    suppressing an entire object (e.g. a cat body).

    The ``area_ratio`` guard ensures that the containment swap only
    triggers for true containment (e.g. eye 2 k inside body 90 k),
    not for similarly-sized overlapping masks.

    Args:
        iom_threshold: IoM threshold above which a lower-scored mask
            is suppressed. Default: ``0.3``.
        score_margin: Maximum score difference below which the larger
            containing mask is preferred over the smaller contained one.
            ``None`` disables containment-aware logic. Default: ``None``.
        area_ratio: Maximum winner/loser area ratio for the containment
            swap.  Default: ``0.5``.
    """

    def __init__(
        self,
        iom_threshold: float = 0.3,
        score_margin: float | None = None,
        area_ratio: float = 0.5,
    ) -> None:
        """Initialize BoxIoMNMS with the given IoM threshold."""
        super().__init__()
        self.iom_threshold = iom_threshold
        self.score_margin = score_margin
        self.area_ratio = area_ratio

    def forward(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply box-IoM NMS.

        Args:
            masks: Binary masks ``[N, H, W]``.
            scores: Confidence scores ``[N]``.
            labels: Category labels ``[N]``.

        Returns:
            Filtered (masks, scores, labels).
        """
        if masks.size(0) <= 1:
            return masks, scores, labels

        boxes = masks_to_boxes_traceable(masks)
        iom_matrix = _pairwise_box_iom(boxes.float())
        areas = masks.bool().flatten(1).sum(dim=1).float() if self.score_margin is not None else None
        keep = _greedy_nms(
            scores,
            iom_matrix,
            self.iom_threshold,
            areas=areas,
            score_margin=self.score_margin,
            area_ratio=self.area_ratio,
        )
        return masks[keep], scores[keep], labels[keep]


class MaskIoMNMS(PostProcessor):
    """Non-Maximum Suppression using mask IoM (Intersection over Minimum).

    Better than IoU-based NMS for nested/contained objects. A small mask
    fully inside a larger one gets IoM ≈ 1.0 and will be suppressed,
    whereas IoU-based NMS would keep it.

    Supports **containment-aware** mode via ``score_margin``.  When the
    higher-scored mask is *contained inside* a lower-scored mask and the
    score gap is within the margin, the larger mask is preferred.  This
    prevents tiny high-confidence fragments (e.g. an eye) from
    suppressing an entire object (e.g. a cat body).

    The ``area_ratio`` guard ensures that the containment swap only
    triggers for true containment (e.g. eye 2 k inside body 90 k),
    not for similarly-sized overlapping masks.

    Adapted from SAM3's ``remove_overlapping_masks`` approach.

    Args:
        iom_threshold: IoM threshold above which a lower-scored mask
            is suppressed. Default: ``0.3``.
        score_margin: Maximum score difference below which the larger
            containing mask is preferred over the smaller contained one.
            ``None`` disables containment-aware logic. Default: ``None``.
        area_ratio: Maximum winner/loser area ratio for the containment
            swap.  Default: ``0.5``.
    """

    def __init__(
        self,
        iom_threshold: float = 0.3,
        score_margin: float | None = None,
        area_ratio: float = 0.5,
    ) -> None:
        """Initialize MaskIoMNMS with the given IoM threshold."""
        super().__init__()
        self.iom_threshold = iom_threshold
        self.score_margin = score_margin
        self.area_ratio = area_ratio

    def forward(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply mask-IoM NMS.

        Args:
            masks: Binary masks ``[N, H, W]``.
            scores: Confidence scores ``[N]``.
            labels: Category labels ``[N]``.

        Returns:
            Filtered (masks, scores, labels).
        """
        if masks.size(0) <= 1:
            return masks, scores, labels

        iom_matrix = _pairwise_mask_iom(masks.bool())
        areas = masks.bool().flatten(1).sum(dim=1).float() if self.score_margin is not None else None
        keep = _greedy_nms(
            scores,
            iom_matrix,
            self.iom_threshold,
            areas=areas,
            score_margin=self.score_margin,
            area_ratio=self.area_ratio,
        )
        return masks[keep], scores[keep], labels[keep]


class SoftNMS(PostProcessor):
    """Soft-NMS with Gaussian score decay on mask IoU.

    Instead of hard suppression, scores are decayed proportionally
    to overlap::

        s_i <- s_i * exp(-IoU^2 / sigma)

    Masks whose decayed score falls below ``score_threshold`` are removed.
    This is gentler than hard NMS and works better when nearby distinct
    objects have overlapping masks.

    Reference: "Soft-NMS -- Improving Object Detection With One Line of Code"
    https://arxiv.org/abs/1704.04503


    Args:
        sigma: Gaussian decay parameter. Smaller values = more aggressive
            suppression. Default: ``0.5``.
        score_threshold: Minimum score after decay. Default: ``0.1``.
    """

    def __init__(self, sigma: float = 0.5, score_threshold: float = 0.1) -> None:
        """Initialize SoftNMS with sigma and score threshold."""
        super().__init__()
        self.sigma = sigma
        self.score_threshold = score_threshold

    def forward(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Soft-NMS with Gaussian decay on mask IoU.

        Args:
            masks: Binary masks ``[N, H, W]``.
            scores: Confidence scores ``[N]``.
            labels: Category labels ``[N]``.

        Returns:
            Filtered (masks, scores, labels) with decayed scores.
        """
        if masks.size(0) <= 1:
            return masks, scores, labels

        iou_matrix = _pairwise_mask_iou(masks.bool())
        n = masks.size(0)
        decayed_scores = scores.clone().float()

        # Process in order of descending score
        order = torch.argsort(decayed_scores, descending=True)

        for i in range(n):
            idx = order[i]
            # Decay scores of all subsequent (lower-priority) masks
            for j in range(i + 1, n):
                jdx = order[j]
                iou = iou_matrix[idx, jdx]
                decay = torch.exp(-(iou * iou) / self.sigma)
                decayed_scores[jdx] *= decay

        keep = decayed_scores > self.score_threshold
        return masks[keep], decayed_scores[keep], labels[keep]
