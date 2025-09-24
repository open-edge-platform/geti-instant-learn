# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Convert masks to polygons."""

import cv2
import numpy as np

from getiprompt.processes.mask_processors.mask_processor_base import (
    MaskProcessor,
)
from getiprompt.types import Annotations, Masks


class MasksToPolygons(MaskProcessor):
    """This class converts a list of masks to a list of annotations (polygons).

    Examples:
        >>> from getiprompt.processes.mask_processors import MasksToPolygons
        >>> from getiprompt.types import Masks
        >>> import torch
        >>>
        >>> processor = MasksToPolygons()
        >>>
        >>> # Create a simple square mask.
        >>> mask_tensor = torch.zeros((1, 10, 10), dtype=torch.bool)
        >>> mask_tensor[0, 2:8, 2:8] = True
        >>> sample_mask = Masks()
        >>> sample_mask.add(mask_tensor, class_id=0)
        >>>
        >>> annotations = processor([sample_mask])
        >>>
        >>> # The output should be a single annotation object with one polygon.
        >>> len(annotations)
        1
        >>> len(annotations[0].polygons[0])
        1
        >>> # The polygon should have 4 points (a square).
        >>> len(annotations[0].polygons[0][0])
        4
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, masks: list[Masks] | None = None) -> list[Annotations]:
        """Convert a list of masks to a list of annotations (polygons)."""
        annotations_list = []

        for mask_obj in masks:
            annotation = Annotations()

            for class_id in mask_obj.data:
                instance_masks = mask_obj.data[class_id].cpu().numpy()

                for instance_idx in range(len(instance_masks)):
                    mask = instance_masks[instance_idx].astype(np.uint8) * 255
                    contours, _ = cv2.findContours(
                        mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )

                    for contour in contours:
                        # Simplify the contour to reduce number of points
                        epsilon = 0.005 * cv2.arcLength(contour, closed=True)
                        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

                        # Convert to list of [x, y] coordinates
                        polygon = approx.reshape(-1, 2).tolist()

                        # Only add polygons with at least 3 points
                        if len(polygon) >= 3:
                            annotation.add_polygon(polygon, class_id)

            annotations_list.append(annotation)

        return annotations_list
