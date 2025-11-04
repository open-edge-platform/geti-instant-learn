# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from getiprompt.data.base.sample import Sample


class MultiInstancePriorFilter(nn.Module):
    """Filter out large boxes that are mostly covered by smaller boxes.

    A box is filtered if the combined area of all smaller boxes fully contained
    within it exceeds the threshold of its own area.
    """

    def __init__(self, threshold: float = 0.8) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, pred_samples: list[Sample]) -> list[Sample]:
        """Filter the boxes based on containment and area ratio.

        Args:
            priors: A list of Priors objects, one for each image.

        Returns:
            A list of Priors objects with the large container boxes filtered out.
        """
        boxes_per_image = []
        for pred in pred_samples:
            if pred.bboxes is None:
                continue

            cls_boxes: dict[int, list[torch.Tensor]] = {}
            for class_id, box, score in zip(pred.category_ids, pred.bboxes, pred.scores, strict=False):
                if class_id not in cls_boxes:
                    cls_boxes[class_id] = []
                cls_boxes[class_id].append(torch.cat((box, score.unsqueeze(0))))

            for class_id, boxes in cls_boxes.items():
                original_box_tensor = torch.stack(boxes)
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
                    cls_boxes[class_id] = original_box_tensor[final_indices]
            boxes_per_image.append(cls_boxes)
        return boxes_per_image
