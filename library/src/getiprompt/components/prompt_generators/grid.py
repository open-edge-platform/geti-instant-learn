# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Grid-based prompt generator."""

import torch
from torchvision import tv_tensors

from getiprompt.components.prompt_generators.base import PromptGenerator
from getiprompt.types import Priors, Similarities


class GridPromptGenerator(PromptGenerator):
    """This class generates prompts for the segmenter.

    This is based on the similarities between the reference and target images.

    Examples:
        >>> import torch
        >>> from getiprompt.processes.prompt_generators import GridPromptGenerator
        >>> from getiprompt.types import Similarities
        >>>
        >>> prompt_generator = GridPromptGenerator(num_grid_cells=2)
        >>> similarities = Similarities()
        >>> # Create a similarity map with a clear hot-spot
        >>> sim_map = torch.zeros(1, 10, 10)
        >>> sim_map[0, 2:4, 2:4] = 0.8
        >>> similarities.add(sim_map, class_id=1)
        >>> image = tv_tensors.Image(torch.zeros(3, 20, 20))
        >>> image.add_local_features(image.global_features[:6], 1)
        >>> priors = prompt_generator(target_similarities=[similarities], target_images=[image])
        >>> isinstance(priors[0], Priors) and priors[0].points.get(1) is not None
        True
        >>> result_points = priors[0].points.get(1)
        >>> result_points is not None and len(result_points) > 0
        True
    """

    def __init__(
        self,
        num_grid_cells: int = 16,
        similarity_threshold: float = 0.65,
        num_bg_points: int = 1,
    ) -> None:
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

    def forward(
        self,
        target_similarities: list[Similarities] | None = None,
        target_images: list[tv_tensors.Image] | None = None,
    ) -> list[Priors]:
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
            List[Priors] List of priors, one per target image instance
        """
        priors_per_image: list[Priors] = []

        if target_similarities is None:
            target_similarities = [Similarities()]
        if target_images is None:
            target_images = [tv_tensors.Image()]
for similarities_per_image, target_image in zip(target_similarities, target_images, strict=True):
            priors = Priors()
            original_image_shape = target_image.shape[-2:]  # (width, height)

            for class_id, class_similarity_maps in similarities_per_image.data.items():
                background_points_enc = self._get_background_points(class_similarity_maps)  # Operates on (H_enc, W_enc)

                # Convert background points to original image coordinates
                background_points_orig = self._convert_points_to_original_size(
                    background_points_enc,
                    class_similarity_maps.shape[-2:],  # input_map_shape (H_map, W_map)
                    original_image_shape,  # original_image_size (W_orig, H_orig)
                )

                for similarity_map_enc in class_similarity_maps:  # Each map is (H_map, W_map)
                    foreground_points_enc = self._get_foreground_points(similarity_map_enc)

                    # Skip if no foreground points found
                    if len(foreground_points_enc) == 0:
                        priors.points.add(
                            torch.empty((0, 4), device=similarity_map_enc.device),
                            class_id,
                        )
                        continue

                    foreground_points_orig = self._convert_points_to_original_size(
                        foreground_points_enc,
                        similarity_map_enc.shape,  # input_map_shape (H_map, W_map)
                        original_image_shape,  # original_image_size (W_orig, H_orig)
                    )

                    fg_point_labels = torch.ones(
                        (len(foreground_points_orig), 1),
                        device=foreground_points_orig.device,
                    )
                    bg_point_labels = torch.zeros(
                        (len(background_points_orig), 1),
                        device=background_points_orig.device,
                    )

                    all_points = torch.cat(
                        [
                            torch.cat([foreground_points_orig, fg_point_labels], dim=1),
                            torch.cat([background_points_orig, bg_point_labels], dim=1),
                        ],
                        dim=0,
                    )
                    priors.points.add(all_points, class_id)

            priors = self._filter_duplicate_points(priors)
            priors_per_image.append(priors)
        return priors_per_image

    def _get_foreground_points(
        self,
        similarity: torch.Tensor,
    ) -> torch.Tensor:
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

        bg_coords = torch.stack((bg_x_coords, bg_y_coords, bg_values), dim=0).T  # (N, 3)
        return bg_coords.float()
