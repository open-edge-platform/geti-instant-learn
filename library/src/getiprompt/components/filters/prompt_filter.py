# Copyright (C) 2025 Intel Corporation
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
        >>> from getiprompt.components.filters.prompt_filter import BoxPromptFilter
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

    def _filter_single_category(self, boxes: torch.Tensor, n_valid: int) -> tuple[torch.Tensor, int]:
        """Filter boxes for a single category.

        Args:
            boxes: Box tensor [max_boxes, 5] with (x1, y1, x2, y2, score)
            n_valid: Number of valid boxes

        Returns:
            Filtered boxes tensor and new valid count
        """
        if n_valid == 0:
            return boxes, 0

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
            return boxes, 0

        # Get original indices to keep and sort them
        original_indices_to_keep = sorted_indices[keep_mask]
        final_indices, _ = torch.sort(original_indices_to_keep)

        # Create new filtered output
        filtered_boxes = boxes.clone()
        num_kept = len(final_indices)
        filtered_boxes[:num_kept] = valid_boxes[final_indices]
        filtered_boxes[num_kept:] = 0  # Zero out unused slots

        return filtered_boxes, num_kept

    def forward(
        self,
        box_prompts: torch.Tensor,
        num_boxes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Filter boxes based on containment and area ratio.

        Args:
            box_prompts: Box prompts [T, C, max_boxes, 5]
            num_boxes: Number of valid boxes [T, C]

        Returns:
            Filtered box_prompts [T, C, max_boxes, 5] and updated num_boxes [T, C]
        """
        filtered_prompts = box_prompts.clone()
        filtered_num = num_boxes.clone()

        num_images, num_categories = box_prompts.shape[:2]

        for img_idx in range(num_images):
            for cat_idx in range(num_categories):
                n_valid = int(num_boxes[img_idx, cat_idx].item())
                boxes = box_prompts[img_idx, cat_idx]

                filtered_boxes, new_count = self._filter_single_category(boxes, n_valid)
                filtered_prompts[img_idx, cat_idx] = filtered_boxes
                filtered_num[img_idx, cat_idx] = new_count

        return filtered_prompts, filtered_num


class PointPromptFilter(nn.Module):
    """Point prompt filter that reduces the number of points in points_per_image to a maximum value.

    This selects the points with the highest scores for each class.

    Args:
        num_foreground_points: Maximum number of foreground points to keep per class

    Example:
        >>> import torch
        >>> from getiprompt.components.filters.prompt_filter import PointPromptFilter
        >>> filter = PointPromptFilter(num_foreground_points=2)
        >>> points_per_image = {1: torch.tensor([
        ...     [10, 10, 0.9, 1],
        ...     [20, 20, 0.8, 1],
        ...     [30, 30, 0.7, 1],
        ... ])
        >>> bg_points_1 = torch.tensor([
        ...     [1, 1, 0.1, 0],
        ...     [2, 2, 0.2, 0],
        ... ])
        >>> point_prompts = {1: torch.cat([fg_points_1, bg_points_1])}
        >>> filtered_point_prompts = filter(point_prompts)
        >>> len(filtered_point_prompts[1])
        2
    """

    def __init__(self, num_foreground_points: int = 40) -> None:
        """Initialize the point prompt filter."""
        super().__init__()
        self.num_foreground_points = num_foreground_points

    def _filter_points(self, points: torch.Tensor) -> torch.Tensor:
        """Filter a single list of points based on scores. This method adds all background points.

        Args:
            points: Tensor of points with shape (N, 4) where each row is [x, y, score, label]

        Returns:
            Filtered points tensor
        """
        # If points is empty or fewer than max_num_points, return as is
        if points.shape[0] <= self.num_foreground_points:
            return points

        fg_indices = (points[:, 3] == 1).nonzero()[:, 0]
        bg_indices = (points[:, 3] == 0).nonzero()[:, 0]

        fg_points = points[fg_indices]
        bg_points = points[bg_indices]

        _, fg_indices_sorted = torch.sort(fg_points[:, 2], descending=True)
        fg_indices_select = fg_indices_sorted[: self.num_foreground_points]
        fg_points_select = fg_points[fg_indices_select]

        # return best matching foreground points and add all background_points
        return torch.cat([fg_points_select, bg_points])

    def forward(self, prompts_per_image: list[dict[int, torch.Tensor]]) -> list[dict[int, torch.Tensor]]:
        """Filter points in the prompts, keeping the ones with the highest scores.

        Args:
            prompts_per_image(list[dict[int, torch.Tensor]]): List of prompts per image, one per target image instance.

        Returns:
            prompts_per_image(list[dict[int, torch.Tensor]]):
                List of prompts per image, one per target image instance, with the points filtered.
        """
        for class_prompts in prompts_per_image:
            for class_id, points in class_prompts.items():
                filtered_points = self._filter_points(points)
                class_prompts[class_id] = filtered_points

        return prompts_per_image
