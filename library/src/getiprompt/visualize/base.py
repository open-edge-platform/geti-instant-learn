# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for visualization processes."""

from getiprompt.types import Boxes, Masks, Points, Priors


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
    def masks_from_priors(priors: list[Priors]) -> list[Masks]:
        """Converts a list of shape 1HW to a List of Masks.

        Note: The first channel of arrays contains instance ids in range [0..max()].

        Args:
            priors: The list of priors to convert

        Returns:
            The list of masks
        """
        return [m.masks for m in priors]

    @staticmethod
    def boxes_from_priors(priors: list[Priors]) -> list[Boxes]:
        """Extracts boxes from priors.

        Args:
            priors: The list of priors to extract boxes from

        Returns:
            The list of boxes
        """
        return [m.boxes for m in priors]

    @staticmethod
    def points_from_priors(priors: list[Priors]) -> list[Points]:
        """Extracts points from priors.

        Args:
            priors: The list of priors to extract points from

        Returns:
            The list of points
        """
        return [m.points for m in priors]

    def __call__(self) -> None:
        """Call visualization process."""
