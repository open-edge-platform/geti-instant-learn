# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Resize masks."""

from getiprompt.processes import Process
from getiprompt.types import Priors


class ResizeMasks(Process):
    """This process resizes the masks to the given size.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from getiprompt.processes.preprocessors import ResizeMasks
        >>> from getiprompt.types import Priors, Masks
        >>>
        >>> resizer = ResizeMasks(size=(10, 20))
        >>> masks = Masks()
        >>> masks.add(np.zeros((100, 100), dtype=np.uint8))
        >>> priors = Priors(masks=masks)
        >>> resized_priors = resizer([priors])
        >>> resized_priors[0].masks.shape
        torch.Size([1, 20, 10])
    """

    def __init__(self, size: int | tuple[int, int] | None = None) -> None:
        """This initializes the process.

        Args:
            size: The size to resize the masks to. If a tuple is provided, the masks will be resized to the given width
              and height. If an integer is provided, the masks will be resized to the given size, maintaining aspect
                ratio. If None is provided, the masks will not be resized.
        """
        super().__init__()
        self.size = size

    def __call__(
        self,
        priors: list[Priors],
    ) -> list[Priors]:
        """Inspect overlapping areas between different label masks.

        Args:
            priors: List of Priors objects of which the masks will be resized

        Returns:
            List of Priors objects with resized masks
        """
        for prior in priors:
            prior.masks.resize_inplace(self.size)
        return priors
