# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for visualization processes."""

import numpy as np
import torch

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

    @staticmethod
    def binary_masks_to_masks(
        arrays: list[np.ndarray | None],
        class_id: int = 0,
    ) -> list[Masks]:
        """Converts binary mask arrays to Masks objects.

        This method is designed for GetiPromptBatch.masks_np which provides
        binary masks already separated by instance.

        Args:
            arrays: List of arrays with shape (N, H, W) containing binary masks,
                   or None for samples without masks
            class_id: The class id to use for all masks

        Returns:
            List of Masks objects

        Example:
            >>> batch = next(iter(dataloader))
            >>> gt_masks = visualizer.binary_masks_to_masks(batch.masks_np)
            >>> # For PerSeg: each array has shape (1, H, W)
            >>> # For LVIS: each array may have shape (N, H, W) where N > 1
        """
        masks_list = []
        for mask_array in arrays:
            if mask_array is None:
                # Create empty Masks object for samples without masks
                masks_list.append(Masks())
            else:
                # mask_array has shape (N, H, W) - already binary masks per instance
                masks_obj = Masks()
                for instance_idx in range(mask_array.shape[0]):
                    # Add each instance mask with the same class_id
                    masks_obj.add(mask_array[instance_idx], class_id=class_id)
                masks_list.append(masks_obj)
        return masks_list

    def __call__(self) -> None:
        """Call visualization process."""
