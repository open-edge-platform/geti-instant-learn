# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This module provides a filter that choces the top K foreground points."""

import torch
from torch import nn


class PointFilter(nn.Module):
    """Point filter that reduces the number of points in points_per_image to a maximum value.

    This selects the points with the highest scores for each class.

    Example:
        >>> import torch
        >>> from getiprompt.components.filters.max_point_filter import PointFilter
        >>> filter = PointFilter(max_num_points=2)
        >>> points_per_image = {1: torch.tensor([
        ...     [10, 10, 0.9, 1],
        ...     [20, 20, 0.8, 1],
        ...     [30, 30, 0.7, 1],
        ... ])
        >>> bg_points_1 = torch.tensor([
        ...     [1, 1, 0.1, 0],
        ...     [2, 2, 0.2, 0],
        ... ])
        >>> points_1 = torch.cat([fg_points_1, bg_points_1])
        >>> priors.points.add(points_1, class_id=1)
        >>> filtered_priors_list = filter([priors])
        >>> filtered_priors = filtered_priors_list[0]
        >>> points_per_class = filtered_priors.points.get(1)
        >>> len(points_per_class)
        1
        >>> filtered_points_1 = points_per_class[0]
        >>> len(filtered_points_1)
        4
        >>> int((filtered_points_1[:, 3] == 1).sum())
        2
    """

    def __init__(self, num_foreground_points: int = 40) -> None:
        """Initialize the max point filter.

        Args:
            num_foreground_points: Maximum number of foreground points to keep per class
        """
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
