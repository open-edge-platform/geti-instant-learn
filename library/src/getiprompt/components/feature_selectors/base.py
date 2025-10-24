# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for feature selectors."""

from abc import ABC, abstractmethod

import torch
from torch import nn

from getiprompt.types import Features


class FeatureSelector(nn.Module, ABC):
    """This is the base class for feature selectors.

    Examples:
        >>> from getiprompt.processes.feature_selectors import FeatureSelector
        >>> from getiprompt.types import Features
        >>>
        >>> class MyFeatureSelector(FeatureSelector):
        ...     def __call__(self, features_per_image: list[Features]) -> list[Features]:
        ...         return []
        >>>
        >>> my_selector = MyFeatureSelector()
        >>> features = my_selector([Features()])
    """

    @staticmethod
    def get_all_local_class_features(
        features_per_image: list[Features],
    ) -> dict[int, list[torch.Tensor]]:
        """This method gets all features for all classes over all images.

        Args:
            features_per_image: A list of features for each reference image.

        Returns:
            A dictionary of features for each class.

        Examples:
            >>> import torch
            >>> from getiprompt.types import Features
            >>> from getiprompt.processes.feature_selectors import FeatureSelector
            >>> features1 = Features()
            >>> features1.local_features={
            ...         1: [torch.randn(1, 4), torch.randn(1, 4)],
            ...         2: [torch.randn(1, 4)],
            ...     }
            >>> features2 = Features()
            >>> features2.local_features={
            ...         1: [torch.randn(1, 4)],
            ...         3: [torch.randn(1, 4), torch.randn(1, 4)],
            ...     }
            >>> all_features = FeatureSelector.get_all_local_class_features([features1, features2])
            >>> sorted(all_features.keys())
            [1, 2, 3]
            >>> len(all_features[1])
            3
            >>> len(all_features[2])
            1
            >>> len(all_features[3])
            2
        """
        all_features_per_class = {}

        # First collect all features per class over all images
        for features in features_per_image:
            for class_id, local_features_list in features.local_features.items():
                if class_id not in all_features_per_class:
                    all_features_per_class[class_id] = []
                all_features_per_class[class_id].extend(local_features_list)

        return all_features_per_class

    @abstractmethod
    def forward(self, features_per_image: list[Features]) -> list[Features]:
        """Forward method that must be implemented by subclasses.

        Args:
            features_per_image: A list of features for each reference image.

        Returns:
            A list of processed features.
        """
