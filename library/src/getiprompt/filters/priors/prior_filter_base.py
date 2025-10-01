# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for prior filters."""

from getiprompt.filters import Filter
from getiprompt.types import Priors


class PriorFilter(Filter):
    """This is the base class for all prior filters.

    Example:
        >>> import torch
        >>> from getiprompt.types import Priors
        >>> filter = PriorFilter()
        >>> priors = Priors()
        >>> priors.points.add(torch.tensor([[1, 2, 0.5, 1]]), class_id=1)
        >>> filtered_priors = filter([priors])
        >>> isinstance(filtered_priors, list)
        True
    """

    def __call__(self, priors: list[Priors]) -> list[Priors]:
        """Filter the priors."""
        return priors
