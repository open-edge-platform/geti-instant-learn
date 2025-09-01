# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for segmenters."""

from abc import abstractmethod

from getiprompt.processes import Process
from getiprompt.types import Image, Masks


class Segmenter(Process):
    """This class extracts segmentation masks.

    Examples:
        >>> from getiprompt.processes.segmenters import Segmenter
        >>> from getiprompt.types import Image, Masks
        >>> import numpy as np
        >>>
        >>> class MySegmenter(Segmenter):
        ...     def __call__(self, images: list[Image], **kwargs) -> list[Masks]:
        ...         return [Masks() for _ in images]
        >>>
        >>> my_segmenter = MySegmenter()
        >>> sample_image = np.zeros((10, 10, 3), dtype=np.uint8)
        >>> masks = my_segmenter([Image(sample_image)])
        >>> len(masks)
        1
        >>> isinstance(masks[0], Masks)
        True
    """

    @abstractmethod
    def __call__(self, images: list[Image]) -> list[Masks]:
        """This method extracts segmentation masks.

        Args:
            images: The images to segment.

        Returns:
            Segmentation masks.
        """
