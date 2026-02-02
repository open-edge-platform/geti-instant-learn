# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This module provides filters for box and point prompts."""

import torch
from torch import nn


class BoxPromptFilter(nn.Module):
    """Filter out large boxes that are mostly covered by smaller boxes.

    A box is filtered if the combined area of all smaller boxes fully contained
    within it exceeds the threshold of its own area.

    All inputs and outputs are tensors for full traceability.

    Args:
        threshold: Threshold for the box prompt filter.

    Example:
        >>> import torch
        >>> from instantlearn.components.filters.prompt_filter import BoxPromptFilter
        >>> filter = BoxPromptFilter(threshold=0.8)
        >>> # box_prompts: [T, C, max_boxes, 5], num_boxes: [T, C]
        >>> box_prompts = torch.zeros(1, 2, 10, 5)
        >>> box_prompts[0, 0, 0] = torch.tensor([10, 10, 100, 100, 0.9])  # Large box
        >>> box_prompts[0, 0, 1] = torch.tensor([20, 20, 80, 80, 0.8])    # Contained box
        >>> num_boxes = torch.tensor([[2, 0]])
        >>> filtered_prompts, filtered_num = filter(box_prompts, num_boxes)
    """

    def __init__(self, threshold: float = 0.8) -> None:
        """Initialize the box prompt filter."""
        super().__init__()
        self.threshold = threshold

    def _filter_single_category(self, boxes: torch.Tensor) -> torch.Tensor:
        """Filter boxes for a single category.

        Args:
            boxes: Box tensor [max_boxes, 5] with (x1, y1, x2, y2, score)

        Returns:
            Filtered boxes tensor and new valid count
        """
        if (boxes[:, -1] != 0).sum() == 0:
            return boxes
        n_valid = (boxes[:, -1] != 0).sum().item()
        valid_boxes = boxes[:n_valid]
        x1, y1, x2, y2 = valid_boxes[:, 0], valid_boxes[:, 1], valid_boxes[:, 2], valid_boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # Sort boxes by area in descending order
        sorted_indices = torch.argsort(areas, descending=True)
        sorted_areas = areas[sorted_indices]
        sorted_x1, sorted_y1, sorted_x2, sorted_y2 = (
            x1[sorted_indices],
            y1[sorted_indices],
            x2[sorted_indices],
            y2[sorted_indices],
        )

        # Create containment matrix
        is_contained = (
            (sorted_x1[None, :] >= sorted_x1[:, None])
            & (sorted_y1[None, :] >= sorted_y1[:, None])
            & (sorted_x2[None, :] <= sorted_x2[:, None])
            & (sorted_y2[None, :] <= sorted_y2[:, None])
        )
        torch.diagonal(is_contained).fill_(value=False)

        sum_of_contained_areas = torch.sum(is_contained.float() * sorted_areas[None, :], dim=1)
        keep_mask = sum_of_contained_areas <= (self.threshold * (sorted_areas + 1e-9))

        if not keep_mask.any():
            return boxes

        # Get original indices to keep and sort them
        original_indices_to_keep = sorted_indices[keep_mask]
        final_indices, _ = torch.sort(original_indices_to_keep)

        # Create new filtered output
        filtered_boxes = boxes.clone()
        num_kept = len(final_indices)
        filtered_boxes[:num_kept] = valid_boxes[final_indices]
        filtered_boxes[num_kept:] = 0  # Zero out unused slots

        return filtered_boxes

    def forward(
        self,
        box_prompts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Filter boxes based on containment and area ratio.

        Args:
            box_prompts: Box prompts [T, C, max_boxes, 5]

        Returns:
            Filtered box_prompts [T, C, max_boxes, 5] and updated num_boxes [T, C]
        """
        filtered_prompts = box_prompts.clone()
        num_images, num_categories = box_prompts.shape[:2]
        for img_idx in range(num_images):
            for cat_idx in range(num_categories):
                boxes = box_prompts[img_idx, cat_idx]
                filtered_boxes = self._filter_single_category(boxes)
                filtered_prompts[img_idx, cat_idx] = filtered_boxes
        return filtered_prompts
