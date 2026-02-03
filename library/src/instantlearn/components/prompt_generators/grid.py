# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Grid-based prompt generator."""

import torch
from torch import nn


class GridPromptGenerator(nn.Module):
    """Generates point prompts for segmentation using grid-based sampling on similarity maps.

    This prompt generator divides similarity maps into grid cells and selects the best
    scoring point from each cell, enabling multi-object detection in target images.

    All outputs are tensors for full traceability (ONNX/TorchScript compatible).

    Args:
        num_grid_cells: Number of grid cells along each dimension. Default: 16.
        point_selection_threshold: Minimum feature similarity for selecting
            foreground points. Higher values = fewer, more confident point
            proposals. Default: 0.65.
        num_bg_points: Number of background points to sample. Default: 1.
        num_foreground_points: Maximum foreground points to keep per category. Default: 40.
        max_points: Maximum total points per category for output padding. Default: 42.

    Examples:
        >>> import torch
        >>> from instantlearn.components.prompt_generators import GridPromptGenerator
        >>>
        >>> generator = GridPromptGenerator(num_grid_cells=2)
        >>> # Similarities: [T, C, H, W]
        >>> similarities = torch.zeros(1, 2, 10, 10)
        >>> similarities[0, 0, 2:4, 2:4] = 0.8  # Hot-spot for category 0
        >>> category_ids = [1, 2]
        >>> original_sizes = torch.tensor([[20, 20]])
        >>>
        >>> point_prompts, num_points = generator(similarities, category_ids, original_sizes)
        >>> point_prompts.shape  # [T, C, max_points, 4]
        torch.Size([1, 2, 42, 4])
        >>> num_points.shape  # [T, C]
        torch.Size([1, 2])
    """

    def __init__(
        self,
        num_grid_cells: int = 16,
        point_selection_threshold: float = 0.65,
        num_bg_points: int = 1,
        num_foreground_points: int = 40,
        max_points: int = 42,
    ) -> None:
        """Initialize the GridPromptGenerator."""
        super().__init__()
        if num_grid_cells <= 0:
            msg = "num_grid_cells must be positive."
            raise ValueError(msg)
        self.num_grid_cells = num_grid_cells
        self.point_selection_threshold = point_selection_threshold
        self.num_bg_points = num_bg_points
        self.num_foreground_points = num_foreground_points
        self.max_points = max_points

    def _get_foreground_points(self, similarity_map: torch.Tensor) -> torch.Tensor:
        """Select foreground points based on the similarity mask and grid-based filtering.

        Operates on the provided similarity map, using self.num_grid_cells to define the grid.

        Args:
            similarity_map: 2D Similarity mask tensor (map_height, map_width)

        Returns:
            Foreground points coordinates and scores with shape (N, 3) where each row is [x, y, score],
            in the input similarity map's coordinate space.
        """
        map_w, map_h = similarity_map.shape

        point_coords = torch.where(similarity_map > self.point_selection_threshold)  # (x_indices, y_indices)
        foreground_coords = torch.stack((point_coords[1], point_coords[0], similarity_map[point_coords]), axis=0).T

        if len(foreground_coords) == 0:
            # Fallback: use the point with the maximum similarity score
            max_idx = similarity_map.argmax()
            max_y = (max_idx // map_h).long()
            max_x = (max_idx % map_h).long()
            max_score = similarity_map[max_y, max_x]
            return torch.tensor([[max_x, max_y, max_score]], device=similarity_map.device, dtype=similarity_map.dtype)

        cell_width = map_w / self.num_grid_cells
        cell_height = map_h / self.num_grid_cells

        if cell_height == 0 or cell_width == 0:
            return foreground_coords[torch.topk(foreground_coords[:, 2], k=1, dim=0, largest=True)[1]]

        # Assign each point to a grid cell ID (row-major order)
        x_coord_on_map = foreground_coords[:, 0]
        y_coord_on_map = foreground_coords[:, 1]
        x_cell_index = (x_coord_on_map / cell_width).floor().long()
        y_cell_index = (y_coord_on_map / cell_height).floor().long()
        x_cell_index = torch.clamp(x_cell_index, 0, self.num_grid_cells - 1)
        y_cell_index = torch.clamp(y_cell_index, 0, self.num_grid_cells - 1)

        idx_grid = (
            y_cell_index * self.num_grid_cells  # Row index * number of columns (which is self.num_grid_cells)
            + x_cell_index  # Column index
        )
        idx_unique_cells = torch.unique(idx_grid)

        selected_points_list = []
        for cell_id in idx_unique_cells:
            points_in_cell_mask = idx_grid == cell_id
            points_in_cell = foreground_coords[points_in_cell_mask]
            if len(points_in_cell) > 0:
                best_point_in_cell = points_in_cell[torch.topk(points_in_cell[:, 2], k=1, dim=0, largest=True)[1]]
                selected_points_list.append(best_point_in_cell)

        points_scores = torch.cat(selected_points_list, dim=0)

        # sort by highest score
        sorted_indices = torch.argsort(points_scores[:, -1], descending=True)
        return points_scores[sorted_indices]

    def _get_background_points(self, similarity_map: torch.Tensor) -> torch.Tensor:
        """Select background points based on the similarity mask.

        Operates on the input similarity map (can be 2D or 3D).
        If 3D, sums over the first dimension. Coordinates are relative to the map's H, W.

        Args:
            similarity_map: Similarity mask tensor (H, W) or (num_maps, H, W)

        Returns:
            Background points coordinates with shape (num_bg_points, 3) where each row is [x, y, score]
            in the input similarity map's H, W coordinate space.
        """
        if self.num_bg_points == 0:
            return torch.empty((0, 3), device=similarity_map.device)

        current_similarity_map = similarity_map
        if current_similarity_map.ndim == 3:
            if current_similarity_map.shape[0] == 0:  # Empty stack
                return torch.empty((0, 3), device=similarity_map.device)
            current_similarity_map = current_similarity_map.sum(dim=0)  # Sum over maps

        map_h, map_w = current_similarity_map.shape
        if map_h == 0 or map_w == 0:
            return torch.empty((0, 3), device=similarity_map.device)

        num_elements = current_similarity_map.numel()
        k = min(self.num_bg_points, num_elements)
        if k == 0:
            return torch.empty((0, 3), device=similarity_map.device)

        bg_values, bg_indices_flat = torch.topk(
            current_similarity_map.flatten(),
            k,
            largest=False,
        )

        # Convert flat indices to 2D coordinates (y for rows, x for columns)
        bg_y_coords = (bg_indices_flat // map_w).long()
        bg_x_coords = (bg_indices_flat % map_w).long()

        return torch.stack((bg_x_coords, bg_y_coords, bg_values), dim=0).T.float()  # (N, 3)

    @staticmethod
    def _convert_points_to_original_size(
        input_coords: torch.Tensor,
        input_map_shape: tuple[int, int],
        ori_size: torch.Tensor,
    ) -> torch.Tensor:
        """Convert point coordinates from map space to original image space.

        Args:
            input_coords: Tensor of shape (N, k) with [x, y, ...] coordinates.
            input_map_shape: Tuple (height, width) of the input similarity map.
            ori_size: Tensor [H, W] of the original image size.

        Returns:
            Tensor of shape (N, k) with coordinates scaled to ori_size.
        """
        points_original_coords = input_coords.clone()
        ori_height, ori_width = ori_size[0].item(), ori_size[1].item()
        map_h, map_w = input_map_shape
        if map_w == 0 or map_h == 0:
            return points_original_coords

        scale_x = ori_width / map_w
        scale_y = ori_height / map_h
        points_original_coords[:, 0] *= scale_x
        points_original_coords[:, 1] *= scale_y
        return points_original_coords

    def _filter_foreground_points(self, foreground_points: torch.Tensor) -> torch.Tensor:
        """Filter foreground points to keep only top-scoring ones.

        Args:
            foreground_points: Foreground points [N, 4] with (x, y, score, label)

        Returns:
            Filtered foreground points [M, 4] where M <= num_foreground_points
        """
        if foreground_points.shape[0] <= self.num_foreground_points:
            return foreground_points

        _, top_indices = torch.topk(foreground_points[:, 2], self.num_foreground_points)
        return foreground_points[top_indices]

    def _pad_points(self, points: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Pad points tensor to max_points size.

        Args:
            points: Points tensor [N, 4]
            device: Target device
            dtype: Target dtype

        Returns:
            Padded points tensor [max_points, 4]
        """
        num_points = points.shape[0]
        if num_points >= self.max_points:
            return points[: self.max_points]

        padding = torch.zeros(self.max_points - num_points, 4, device=device, dtype=dtype)
        return torch.cat([points, padding], dim=0)

    def _process_single_category(
        self,
        similarity_map: torch.Tensor,
        original_size: torch.Tensor,
    ) -> torch.Tensor:
        """Process a single category's similarity map to generate points.

        Args:
            similarity_map: Similarity map [H, W]
            original_size: Original image size tensor [H, W]

        Returns:
            Points [N, 4] with (x, y, score, label)
        """
        device = similarity_map.device
        map_shape = similarity_map.shape

        # Get foreground points
        foreground_points = self._get_foreground_points(similarity_map)
        if foreground_points.numel() > 0:
            foreground_points = self._convert_points_to_original_size(foreground_points, map_shape, original_size)
            foreground_labels = torch.ones((foreground_points.shape[0], 1), device=device)
            foreground_points = torch.cat([foreground_points, foreground_labels], dim=1)
            foreground_points = self._filter_foreground_points(foreground_points)
        else:
            foreground_points = torch.empty(0, 4, device=device)

        # Get background points
        background_points = self._get_background_points(similarity_map)
        if background_points.numel() > 0:
            background_points = self._convert_points_to_original_size(background_points, map_shape, original_size)
            background_labels = -torch.ones((background_points.shape[0], 1), device=device)
            background_points = torch.cat([background_points, background_labels], dim=1)
        else:
            background_points = torch.empty(0, 4, device=device)

        return torch.cat([foreground_points, background_points], dim=0)

    def forward(
        self,
        similarities: torch.Tensor,
        category_ids: list[int],
        original_sizes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate point prompts from similarity maps.

        Args:
            similarities: Similarity maps [T, C, H, W]
            category_ids: List of category IDs [C]
            original_sizes: Original image sizes [T, 2] with (H, W)

        Returns:
            point_prompts: [T, C, max_points, 4] - padded point prompts
            num_points: [T, C] - actual valid point counts per (target, category)
        """
        num_targets = similarities.shape[0]
        num_categories = len(category_ids)
        device = similarities.device
        dtype = similarities.dtype

        point_prompts = torch.zeros(num_targets, num_categories, self.max_points, 4, device=device, dtype=dtype)

        for t_idx in range(num_targets):
            original_size = original_sizes[t_idx]

            for c_idx in range(num_categories):
                similarity_map = similarities[t_idx, c_idx]
                points = self._process_single_category(similarity_map, original_size)
                point_prompts[t_idx, c_idx] = self._pad_points(points, device, dtype)

        return point_prompts
