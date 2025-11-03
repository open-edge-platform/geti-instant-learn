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

    def filter_contained_boxes(self, boxes: torch.Tensor, threshold: float = 0.9) -> torch.Tensor:
        """Vectorized containment-based filtering.

        Keeps boxes that don't contain too much total area of smaller boxes inside them.

        Args:
            boxes: A tensor of boxes with shape (N, 4).
                The boxes are expected to be in the format [x1, y1, x2, y2].
            threshold: The threshold for the containment ratio.

        Returns:
            A tensor of filtered boxes with shape (N, 4).
        """
        if boxes.shape[0] <= 1:
            return boxes

        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1) * (y2 - y1)

        # Sort by area descending
        sorted_areas, sorted_idx = areas.sort(descending=True)
        sorted_boxes = boxes[sorted_idx]
        sx1, sy1, sx2, sy2 = sorted_boxes.unbind(1)

        # Containment matrix: is_contained[i, j] = True if box j inside box i
        # Using broadcasting (N, 1) vs (1, N)
        is_contained = (
            (sx1[None, :] >= sx1[:, None])
            & (sy1[None, :] >= sy1[:, None])
            & (sx2[None, :] <= sx2[:, None])
            & (sy2[None, :] <= sy2[:, None])
        )
        is_contained.fill_diagonal_(False)

        # Compute sum of contained areas
        contained_area_sum = (is_contained.float() * sorted_areas[None, :]).sum(dim=1)

        keep_mask = contained_area_sum <= (threshold * (sorted_areas + 1e-9))
        keep_idx = sorted_idx[keep_mask]

        # Restore original order for stable output
        keep_idx, _ = keep_idx.sort()
        return boxes[keep_idx]

    def forward(self, pred_samples: list[Sample]) -> list[dict[int, torch.Tensor]]:
        """Filter the boxes based on containment and area ratio.

        Args:
            pred_samples: A list of Sample objects, one for each image.

        Returns:
            A list of dicts with the large container boxes filtered out.
        """
        boxes_per_image = []
        for pred in pred_samples:
            cls_boxes = {}
            for class_id, box in zip(pred.category_ids, pred.bboxes, strict=False):
                if class_id not in cls_boxes:
                    cls_boxes[class_id] = []
                cls_boxes[class_id].append(box)

            # concat all boxes for each class
            for class_id, boxes in cls_boxes.items():
                boxes = torch.stack(boxes)
                # filtered_boxes = self.filter_contained_boxes(boxes, self.threshold)
                cls_boxes[class_id] = boxes

            boxes_per_image.append(cls_boxes)

        return boxes_per_image
