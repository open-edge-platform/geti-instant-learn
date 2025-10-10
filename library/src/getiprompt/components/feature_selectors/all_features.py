# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Select all features."""

import torch

from getiprompt.components.feature_selectors.base import FeatureSelector
from getiprompt.types import Features


class AllFeaturesSelector(FeatureSelector):
    """This class selects all features over all prior images without averaging.

    Examples:
        >>> import torch
        >>> from getiprompt.processes.feature_selectors import AllFeaturesSelector
        >>> from getiprompt.types import Features
        >>>
        >>> selector = AllFeaturesSelector()
        >>> features1 = Features()
        >>> features1.global_features=torch.randn(1, 4)
        >>> features1.local_features={
        ...         1: [torch.randn(1, 4), torch.randn(1, 4)],
        ...         2: [torch.randn(1, 4)],
        ...     }
        >>> features2 = Features()
        >>> features2.global_features=torch.randn(1, 4)
        >>> features2.local_features={
        ...         1: [torch.randn(1, 4)],
        ...         3: [torch.randn(1, 4), torch.randn(1, 4)],
        ...     }
        >>> all_features = selector([features1, features2])
        >>> len(all_features)
        1
        >>> result = all_features[0]
        >>> result.global_features.shape
        torch.Size([2, 1, 4])
        >>> sorted(result.local_features.keys())
        [1, 2, 3]
        >>> len(result.local_features[1])
        3
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, features_per_image: list[Features]) -> Features:
        """This method merges all features over all prior images without averaging.

        Each class will maintain all its feature vectors from all images.

        Args:
            features_per_image: A list of features for each reference image.

        Returns:
            A Features object containing all features per class.
        """
        result_features = Features()

        # save global features by stacking with extra first dimension
        global_features = torch.cat(
            [image_features.global_features.unsqueeze(0) for image_features in features_per_image],
            dim=0,
        )
        result_features.global_features = global_features
        result_features.local_features = self.get_all_local_class_features(
            features_per_image,
        )

        return result_features
