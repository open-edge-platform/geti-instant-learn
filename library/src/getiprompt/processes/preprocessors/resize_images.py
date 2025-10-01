# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Resize images."""

from getiprompt.processes import Process
from getiprompt.types import Image


class ResizeImages(Process):
    """This process resizes the images to the given size.

    Examples:
        >>> from getiprompt.processes.preprocessors import ResizeImages
        >>> from getiprompt.types import Image
        >>> import numpy as np
        >>>
        >>> resizer = ResizeImages(size=512)
        >>> sample_image = Image(np.zeros((10, 10, 3), dtype=np.uint8))
        >>> resized_images = resizer([sample_image])
        >>> resized_images[0].size
        (512, 512)
    """

    def __init__(self, size: int | tuple[int, int] | None = None) -> None:
        """This initializes the process.

        Args:
            size: The size to resize the images to. If a tuple is provided, the images will be resized to the given
                width and height. If an integer is provided, the images will be resized to the given size, maintaining
                aspect ratio. If None is provided, the images will not be resized.
        """
        super().__init__()
        self.size = size

    def __call__(self, images: list[Image]) -> list[Image]:
        """Resize the images to the given size.

        Args:
            images: The images to resize.


        Returns:
            The resized images.
        """
        for image in images:
            image.resize_inplace(self.size)
        return images
