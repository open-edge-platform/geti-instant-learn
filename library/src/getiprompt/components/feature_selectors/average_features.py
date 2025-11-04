# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Average features."""

import torch

from getiprompt.components.feature_selectors.base import FeatureSelector
from getiprompt.types import Features


class AverageFeatures(FeatureSelector):
    """This class averages features across all reference images and their masks for each class.

    Examples:
        >>> import torch
        >>> from getiprompt.processes.feature_selectors import AverageFeatures
        >>> from getiprompt.types import Features
        >>>
        >>> selector = AverageFeatures()
        >>> features1 = Features()
        >>> features1.local_features={
        ...         1: [torch.ones(1, 4), torch.ones(1, 4)],
        ...         2: [torch.ones(1, 4)],
        ...     }
        >>> features2 = Features()
        >>> features2.local_features={
        ...         1: [torch.ones(1, 4)],
        ...         3: [torch.ones(1, 4), torch.ones(1, 4)],
        ...     }
        >>> avg_features = selector([features1, features2])
        >>> len(avg_features)
        1
        >>> result = avg_features[0]
        >>> sorted(result.local_features.keys())
        [1, 2, 3]
        >>> len(result.local_features[1])
        1
        >>> result.local_features[1][0].shape
        torch.Size([1, 4])
        >>> # Check if the features are correctly averaged and normalized
        >>> torch.allclose(result.local_features[1][0], torch.ones(1, 4) / 2.0)
        True
        >>> torch.allclose(result.local_features[2][0], torch.ones(1, 4) / 2.0)
        True
        >>> torch.allclose(result.local_features[3][0], torch.ones(1, 4) / 2.0)
        True
    """

    def forward(self, features_per_image: list[Features]) -> Features:
        """This method averages all features across all reference images and their masks for each class.

        The result will be a single averaged feature vector per class.

        Args:
            features_per_image: A list of features for each reference image.

        Returns:
            A Features object with the averaged features per class.
        """
        result_features = Features()

        # Average features for each class
        for class_id, feature_list in self.get_all_local_class_features(features_per_image).items():
            stacked_features = torch.cat(feature_list, dim=0)
            averaged_features = stacked_features.mean(dim=0, keepdim=True)
            # allthough features are already normalized, we make sure that the average is normalized too
            averaged_features /= averaged_features.norm(dim=-1, keepdim=True)
            result_features.add_local_features(averaged_features, class_id)

        return result_features
