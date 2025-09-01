# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from getiprompt.filters.priors.prior_filter_base import PriorFilter
from getiprompt.types import Priors
from getiprompt.types.boxes import Boxes


class MultiInstancePriorFilter(PriorFilter):
    """Filter out large boxes that are mostly covered by smaller boxes.

    A box is filtered if the combined area of all smaller boxes fully contained
    within it exceeds the threshold of its own area.
    """

    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold

    def __call__(self, priors: list[Priors]) -> list[Priors]:
        """Filter the boxes based on containment and area ratio.

        Args:
            priors: A list of Priors objects, one for each image.

        Returns:
            A list of Priors objects with the large container boxes filtered out.
        """
        for p in priors:
            if p.boxes is None:
                continue
            boxes = p.boxes
            filtered_boxes = Boxes()
            for class_id in boxes.class_ids():
                class_boxes_list = boxes.get(class_id)
                if not class_boxes_list:
                    continue

                original_box_tensor = torch.cat(class_boxes_list)

                num_boxes = original_box_tensor.shape[0]
                if num_boxes <= 1:
                    filtered_boxes.add(original_box_tensor, class_id=class_id)
                    continue

                x1, y1, x2, y2 = (
                    original_box_tensor[:, 0],
                    original_box_tensor[:, 1],
                    original_box_tensor[:, 2],
                    original_box_tensor[:, 3],
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
                    filtered_boxes.add(original_box_tensor[final_indices], class_id=class_id)
            p.boxes = filtered_boxes
        return priors
