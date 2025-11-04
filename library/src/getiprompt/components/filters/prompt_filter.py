# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This module provides filters for box and point prompts."""

import torch
from torch import nn


class BoxPromptFilter(nn.Module):
    """Filter out large boxes that are mostly covered by smaller boxes.

    A box is filtered if the combined area of all smaller boxes fully contained
    within it exceeds the threshold of its own area.

    Args:
        threshold: Threshold for the box prompt filter.

    Example:
        >>> import torch
        >>> from getiprompt.components.filters.prompt_filter import BoxPromptFilter
        >>> filter = BoxPromptFilter(threshold=0.8)
        >>> boxes_per_image = {1: torch.tensor([
        ...     [10, 10, 20, 20, 0.9, 1],
        ...     [15, 15, 25, 25, 0.8, 1],
        ...     [20, 20, 30, 30, 0.7, 1],
        ... ])
        >>> box_prompts = filter(boxes_per_image)
        >>> len(box_prompts[1])
        1
    """

    def __init__(self, threshold: float = 0.8) -> None:
        """Initialize the box prompt filter."""
        super().__init__()
        self.threshold = threshold

    def forward(self, prompts_per_image: list[dict[int, torch.Tensor]]) -> list[dict[int, torch.Tensor]]:
        """Filter the boxes based on containment and area ratio.

        Args:
            prompts_per_image(list[dict[int, torch.Tensor]]): List of prompts per image, one per target image instance.

        Returns:
            prompts_per_image(list[dict[int, torch.Tensor]]):
                List of prompts per image, one per target image instance, with the large container boxes filtered out.
        """
        for class_prompts in prompts_per_image:
            for class_id, boxes in class_prompts.items():
                x1, y1, x2, y2 = (
                    boxes[:, 0],
                    boxes[:, 1],
                    boxes[:, 2],
                    boxes[:, 3],
                )
                areas = (x2 - x1) * (y2 - y1)

                # Sort boxes by area in descending order. This is for efficient processing of the filter.
                # The `sorted_indices` will be used later to restore the original order.
                sorted_indices = torch.argsort(areas, descending=True)
                sorted_areas = areas[sorted_indices]
                sorted_x1, sorted_y1, sorted_x2, sorted_y2 = (
                    x1[sorted_indices],
                    y1[sorted_indices],
                    x2[sorted_indices],
                    y2[sorted_indices],
                )

                # Create a containment matrix where is_contained[i, j] is True if box j is inside box i.
                # x1[None, :] creates a row vector, and x1[:, None] creates a column vector.
                # The comparison (x1[None, :] >= x1[:, None]) results in a matrix where the (i, j) element
                # is true if box j's x1 is greater than or equal to box i's x1.
                is_contained = (
                    (sorted_x1[None, :] >= sorted_x1[:, None])
                    & (sorted_y1[None, :] >= sorted_y1[:, None])
                    & (sorted_x2[None, :] <= sorted_x2[:, None])
                    & (sorted_y2[None, :] <= sorted_y2[:, None])
                )

                # A box cannot contain itself, so we set the diagonal of the containment matrix to False.
                # This is crucial to avoid adding a box's own area to the sum of contained areas.
                torch.diagonal(is_contained).fill_(value=False)
                # For each box i, sum the areas of all boxes j that are contained within it.
                # `is_contained.float()` converts the boolean matrix to 0s and 1s.
                # `areas[None, :]` broadcasts the areas across the rows.
                sum_of_contained_areas = torch.sum(is_contained.float() * sorted_areas[None, :], dim=1)
                keep_mask = sum_of_contained_areas <= (self.threshold * (sorted_areas + 1e-9))

                if keep_mask.any():
                    # We sort these original indices to ensure the final output preserves the initial order.
                    original_indices_to_keep = sorted_indices[keep_mask]
                    final_indices, _ = torch.sort(original_indices_to_keep)
                    class_prompts[class_id] = boxes[final_indices]
        return prompts_per_image


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
