# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for mask processors."""

from abc import abstractmethod

from getiprompt.processes import Process
from getiprompt.types import Annotations, Masks


class MaskProcessor(Process):
    """This class processes masks to create annotations (polygons).

    Examples:
        >>> from getiprompt.processes.mask_processors import MaskProcessor
        >>> from getiprompt.types import Annotations, Masks
        >>> import torch
        >>>
        >>> # As MaskProcessor is an abstract class, you must subclass it.
        >>> class MyMaskProcessor(MaskProcessor):
        ...     def __call__(self, masks: list[Masks] | None = None) -> list[Annotations]:
        ...         # A real implementation would return an Annotation for each mask.
        ...         return [Annotations()] * len(masks)
        ...
        >>> my_processor = MyMaskProcessor()
        >>> sample_mask = Masks()
        >>> sample_mask.add(torch.zeros((1, 10, 10), dtype=torch.bool), class_id=0)
        >>> annotations = my_processor([sample_mask])
        >>>
        >>> len(annotations)
        1
        >>> isinstance(annotations[0], Annotations)
        True
    """

    @abstractmethod
    def __call__(self, masks: list[Masks] | None = None) -> list[Annotations]:
        """This method extracts polygons from masks.

        Args:
            masks: A list of masks.

        Returns:
            A list of polygons that have been created from the masks.

        """
