# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Grid-based prompt generator."""

import torch
from torchvision import tv_tensors

from getiprompt.components.prompt_generators.base import PromptGenerator
from getiprompt.types import Similarities


class GridPromptGenerator(PromptGenerator):
    """This class generates prompts for the segmenter.

    This is based on the similarities between the reference and target images.

    Examples:
        >>> import torch
        >>> from getiprompt.components.prompt_generators import GridPromptGenerator
        >>> from getiprompt.types import Similarities
        >>>
        >>> prompt_generator = GridPromptGenerator(num_grid_cells=2)
        >>> similarities = Similarities()
        >>> # Create a similarity map with a clear hot-spot
        >>> sim_map = torch.zeros(1, 10, 10)
        >>> sim_map[0, 2:4, 2:4] = 0.8
        >>> similarities.add(sim_map, class_id=1)
        >>> image = tv_tensors.Image(torch.zeros(3, 20, 20))
        >>> point_prompts = prompt_generator(target_similarities=[similarities], target_images=[image])
        >>> isinstance(point_prompts[0], dict) and 1 in point_prompts[0]
        True
        >>> result_points = point_prompts[0][1]
        >>> result_points is not None and len(result_points) > 0
        True
    """

    def __init__(self, num_grid_cells: int = 16, similarity_threshold: float = 0.65, num_bg_points: int = 1) -> None:
        """Generate prompts for the segmenter based on the similarities between the reference and target images.

        Args:
            num_grid_cells: int The number of grid cells to divide the similarity map into, along each dimension.
                                For example, 16 means a 16x16 grid.
            similarity_threshold: float the threshold for the similarity mask
            num_bg_points: int the number of background points to sample

        Raises:
            ValueError: If num_grid_cells is not positive.
        """
        super().__init__()
        if num_grid_cells <= 0:
            msg = "num_grid_cells must be positive."
            raise ValueError(msg)
        self.num_grid_cells = num_grid_cells
        self.similarity_threshold = similarity_threshold
        self.num_bg_points = num_bg_points

    def _get_foreground_points(self, similarity: torch.Tensor) -> torch.Tensor:
        """Select foreground points based on the similarity mask and grid-based filtering.

        Operates on the provided similarity map, using self.num_grid_cells to define the grid.

        Args:
            similarity: 2D Similarity mask tensor (map_height, map_width)

        Returns:
            Foreground points coordinates and scores with shape (N, 3) where each row is [x, y, score],
            in the input similarity map's coordinate space.
        """
        map_w, map_h = similarity.shape

        if map_h == 0 or map_w == 0:
            return torch.empty((0, 3), device=similarity.device)

        point_coords = torch.where(similarity > self.similarity_threshold)  # (x_indices, y_indices)
        foreground_coords = torch.stack(
            (point_coords[1], point_coords[0], similarity[point_coords]),
            axis=0,
        ).T

        if len(foreground_coords) == 0:
            return torch.empty((0, 3), device=similarity.device)

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

        if not selected_points_list:
            return torch.empty((0, 3), device=similarity.device)

        points_scores = torch.cat(selected_points_list, dim=0)

        # sort by highest score
        sorted_indices = torch.argsort(points_scores[:, -1], descending=True)
        return points_scores[sorted_indices]

    def _get_background_points(self, similarity: torch.Tensor) -> torch.Tensor:
        """Select background points based on the similarity mask.

        Operates on the input similarity map (can be 2D or 3D).
        If 3D, sums over the first dimension. Coordinates are relative to the map's H, W.

        Args:
            similarity: Similarity mask tensor (H, W) or (num_maps, H, W)

        Returns:
            Background points coordinates with shape (num_bg_points, 3) where each row is [x, y, score]
            in the input similarity map's H, W coordinate space.
        """
        if self.num_bg_points == 0:
            return torch.empty((0, 3), device=similarity.device)

        current_similarity_map = similarity
        if current_similarity_map.ndim == 3:
            if current_similarity_map.shape[0] == 0:  # Empty stack
                return torch.empty((0, 3), device=similarity.device)
            current_similarity_map = current_similarity_map.sum(dim=0)  # Sum over maps

        map_h, map_w = current_similarity_map.shape
        if map_h == 0 or map_w == 0:
            return torch.empty((0, 3), device=similarity.device)

        num_elements = current_similarity_map.numel()
        k = min(self.num_bg_points, num_elements)
        if k == 0:
            return torch.empty((0, 3), device=similarity.device)

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
    def _filter_duplicate_points(class_point_prompts: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Filter out duplicate points, handling foreground and background points separately.

        Args:
            class_point_prompts: Dictionary with class_id as key and points tensor as value.
                                 Points tensor has shape (N, 4) where columns are [x, y, score, label].

        Returns:
            Dictionary with duplicates removed, keeping highest scoring foreground points
            and lowest scoring background points
        """
        filtered_prompts: dict[int, torch.Tensor] = {}
        for class_id, class_points in class_point_prompts.items():
            if class_points.numel() == 0:
                filtered_prompts[class_id] = class_points
                continue

            # Filter foreground points (keep highest scores)
            foreground_points = class_points[class_points[:, 3] == 1]
            if len(foreground_points) > 0:
                sorted_indices = torch.argsort(foreground_points[:, 2], descending=True)
                foreground_points = foreground_points[sorted_indices]
                _, unique_indices = torch.unique(foreground_points[:, :2], dim=0, return_inverse=True)
                unique_points_foreground = foreground_points[unique_indices]
            else:
                unique_points_foreground = torch.empty((0, 4), device=class_points.device)

            # Filter background points (keep lowest scores)
            background_points = class_points[class_points[:, 3] == 0]
            if len(background_points) > 0:
                sorted_indices = torch.argsort(background_points[:, 2], descending=False)
                background_points = background_points[sorted_indices]
                _, unique_indices = torch.unique(background_points[:, :2], dim=0, return_inverse=True)
                unique_points_background = background_points[unique_indices]
            else:
                unique_points_background = torch.empty((0, 4), device=class_points.device)

            # Combine filtered foreground and background points
            filtered_prompts[class_id] = torch.cat([unique_points_foreground, unique_points_background], dim=0)

        return filtered_prompts

    @staticmethod
    def _convert_points_to_original_size(
        input_coords: torch.Tensor,
        input_map_shape: tuple[int, int],
        ori_size: tuple[int, int],
    ) -> torch.Tensor:
        """Converts point coordinates from an input map's space to original image space.

        Args:
            input_coords: Tensor of shape (N, k) with [x, y, ...] coordinates.
                                   Assumes input_coords[:, 0] is x and input_coords[:, 1] is y.
            input_map_shape: Tuple (height, width) of the input similarity map from which points were derived.
            ori_size: Tuple (width, height) of the original image.

        Returns:
            Tensor of shape (N, k) with [x, y, ...] coordinates scaled to ori_size.
        """
        points_original_coords = input_coords.clone()
        ori_width, ori_height = ori_size
        map_w, map_h = input_map_shape
        if map_w == 0 or map_h == 0:
            return points_original_coords

        scale_x = ori_width / map_w
        points_original_coords[:, 0] *= scale_x
        scale_y = ori_height / map_h
        points_original_coords[:, 1] *= scale_y
        return points_original_coords

    def forward(
        self,
        target_similarities: list[Similarities] | None = None,
        target_images: list[tv_tensors.Image] | None = None,
    ) -> list[dict[int, torch.Tensor]]:
        """This generates prompt candidates (or priors).

        Ths is based on the similarities between the reference and target images.
        It uses a grid based approach to create multi object aware prompt for the segmenter.
        The grid is defined by self.num_grid_cells and applied to the input similarity map's dimensions.

        Args:
            target_similarities: List[Similarities] List of similarities, one per target image instance.
                                Each similarity map within is expected to be 2D (H_map, W_map)
                                or a stack of 2D maps 3D (num_maps, H_map, W_map).
            target_images: List[tv_tensors.Image] List of target image instances

        Returns:
            point_prompts(list[dict[int, torch.Tensor]]):
                List of point prompts (with class_id as key and points as value)
        """
        point_prompts: list[dict[int, torch.Tensor]] = []

        if target_similarities is None:
            target_similarities = [Similarities()]
        if target_images is None:
            target_images = [tv_tensors.Image()]

        for similarities_per_image, target_image in zip(target_similarities, target_images, strict=True):
            class_point_prompts: dict[int, torch.Tensor] = {}
            original_image_shape = target_image.shape[-2:]  # (height, width)

            for class_id, class_similarity_maps in similarities_per_image.data.items():
                background_points = self._get_background_points(class_similarity_maps)  # Operates on (H_enc, W_enc)

                # Convert background points to original image coordinates
                background_points = self._convert_points_to_original_size(
                    background_points,
                    class_similarity_maps.shape[-2:],
                    original_image_shape,
                )

                # Collect all foreground points from all similarity maps for this class
                foreground_points_list = []
                for similarity_map in class_similarity_maps:
                    foreground_points = self._get_foreground_points(similarity_map)

                    # Skip if no foreground points found for this map
                    if len(foreground_points) == 0:
                        continue

                    foreground_points = self._convert_points_to_original_size(
                        foreground_points,
                        similarity_map.shape,
                        original_image_shape,
                    )

                    foreground_labels = torch.ones((len(foreground_points), 1), device=foreground_points.device)
                    foreground_points = torch.cat([foreground_points, foreground_labels], dim=1)
                    foreground_points_list.append(foreground_points)

                # Combine all foreground points from all maps
                if foreground_points_list:
                    foreground_points = torch.cat(foreground_points_list, dim=0)
                else:
                    foreground_points = torch.empty((0, 4)).to(background_points.device)

                # Add background points
                background_labels = torch.zeros((len(background_points), 1), device=background_points.device)
                background_points = torch.cat([background_points, background_labels], dim=1)

                # Combine all points for this class
                class_point_prompts[class_id] = torch.cat([foreground_points, background_points], dim=0)

            # Filter duplicates
            class_point_prompts = self._filter_duplicate_points(class_point_prompts)
            point_prompts.append(class_point_prompts)
        return point_prompts
