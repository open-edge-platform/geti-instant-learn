# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Non-Maximum Suppression post-processors for segmentation masks.

Provides NMS variants operating on mask IoU, box IoU, mask IoM
(Intersection over Minimum), and Soft-NMS with Gaussian score decay.
All implementations use pure PyTorch and are ONNX-exportable.
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


class MaskNMS(PostProcessor):
    """Non-Maximum Suppression using pairwise mask IoU.

    Greedy suppression: iterates masks by descending score, discards
    any mask whose IoU with an already-kept mask exceeds ``iou_threshold``.

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


class MaskIoMNMS(PostProcessor):
    """Non-Maximum Suppression using mask IoM (Intersection over Minimum).

    Better than IoU-based NMS for nested/contained objects. A small mask
    fully inside a larger one gets IoM ≈ 1.0 and will be suppressed,
    whereas IoU-based NMS would keep it.

    Adapted from SAM3's ``remove_overlapping_masks`` approach.

    Args:
        iom_threshold: IoM threshold above which a lower-scored mask
            is suppressed. Default: ``0.3``.
    """

    def __init__(self, iom_threshold: float = 0.3) -> None:
        """Initialize MaskIoMNMS with the given IoM threshold."""
        super().__init__()
        self.iom_threshold = iom_threshold

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
        keep = _greedy_nms(scores, iom_matrix, self.iom_threshold)
        return masks[keep], scores[keep], labels[keep]


class SoftNMS(PostProcessor):
    """Soft-NMS with Gaussian score decay on mask IoU.

    Instead of hard suppression, scores are decayed proportionally
    to overlap::

        s_i <- s_i * exp(-IoU^2 / sigma)

    Masks whose decayed score falls below ``score_threshold`` are removed.
    This is gentler than hard NMS and works better when nearby distinct
    objects have overlapping masks.

    Reference: Bodla et al., "Soft-NMS", ICCV 2017.

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


def _greedy_nms(
    scores: torch.Tensor,
    overlap_matrix: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Greedy NMS using a precomputed overlap matrix.

    Args:
        scores: Confidence scores ``[N]``.
        overlap_matrix: Pairwise overlap values ``[N, N]`` (IoU or IoM).
        threshold: Overlap threshold for suppression.

    Returns:
        Indices of kept masks as a 1-D int64 tensor.
    """
    order = torch.argsort(scores, descending=True)
    kept: list[int] = []

    suppressed = torch.zeros(scores.size(0), dtype=torch.bool, device=scores.device)

    for i in range(scores.size(0)):
        idx = order[i].item()
        if suppressed[idx]:
            continue
        kept.append(idx)
        # Suppress all lower-scored masks that overlap too much
        overlaps = overlap_matrix[idx]
        suppressed |= overlaps > threshold
        suppressed[idx] = False  # never suppress the current mask

    return torch.tensor(kept, dtype=torch.int64, device=scores.device)
