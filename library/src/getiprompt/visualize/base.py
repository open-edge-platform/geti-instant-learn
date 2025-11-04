# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for visualization processes."""

import torch

from getiprompt.types import Boxes, Points


class Visualization:
    """This is the base class for all visualization processes.

    It provides a way to visualize the data.

    Examples:
        >>> from getiprompt.processes.visualizations import Visualization
        >>>
        >>> class MyVisualizer(Visualization):
        ...     def __call__(self, *args, **kwargs):
        ...         pass
        >>>
        >>> my_visualizer = MyVisualizer()
        >>> my_visualizer()
    """

    @staticmethod
    def to_boxes(priors: list[dict[int, torch.Tensor]]) -> list[Boxes]:
        """Extracts boxes from priors.

        Args:
            priors: The list of priors to extract boxes from

        Returns:
            The list of boxes
        """
        boxes_per_image = []
        for cls_boxes in priors:
            boxes_obj = Boxes()
            for class_id, boxes in cls_boxes.items():
                boxes_obj.add(boxes, class_id)
            boxes_per_image.append(boxes_obj)
        return boxes_per_image

    @staticmethod
    def to_points(priors: list[dict[int, torch.Tensor]]) -> list[Points]:
        """Extracts points from priors.

        Args:
            priors: The list of priors to extract points from

        Returns:
            The list of points
        """
        points_per_image = []
        for cls_points in priors:
            points_obj = Points()
            for class_id, points in cls_points.items():
                points_obj.add(points, class_id)
            points_per_image.append(points_obj)
        return points_per_image

    def __call__(self) -> None:
        """Call visualization process."""
