# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from getiprompt.filters import Filter
from getiprompt.types import Features


class FeatureFilter(Filter):
    """This is the base class for all feature filters.

    Examples:
        >>> from getiprompt.filters.features import FeatureFilter
        >>> from getiprompt.types import Features
        >>> import torch
        >>>
        >>> # As FeatureFilter is an abstract class, you must subclass it.
        >>> class MyFeatureFilter(FeatureFilter):
        ...     def __call__(self, features: list[Features]) -> list[Features]:
        ...         # A real implementation would filter the features.
        ...         return features
        ...
        >>> my_filter = MyFeatureFilter()
        >>> sample_features = Features(global_features=torch.zeros((1, 10)))
        >>> result = my_filter([sample_features])
        >>>
        >>> len(result)
        1
        >>> isinstance(result[0], Features)
        True
    """

    def __call__(self, features: list[Features]) -> list[Features]:
        """Filter the features."""
