# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cluster features using KMeans."""

import numpy as np
import torch
from sklearn.cluster import KMeans

from getiprompt.processes.feature_selectors.feature_selector_base import (
    FeatureSelector,
)
from getiprompt.types import Features


class ClusterFeatures(FeatureSelector):
    """This class clusters the features of the reference images.

     This also averages the features per cluster.

    This is loosely based on the paper "Part-aware Personalized Segment Anything Model for
    Patient-Specific Segmentation"
    https://arxiv.org/abs/2403.05433.

    Examples:
        >>> import torch
        >>> from getiprompt.processes.feature_selectors import ClusterFeatures
        >>> from getiprompt.types import Features
        >>>
        >>> selector = ClusterFeatures(num_clusters=2)
        >>> # Create features that are clearly in two clusters
        >>> cluster1 = [torch.ones(1, 4) * i for i in range(1, 5)]
        >>> cluster2 = [torch.ones(1, 4) * i for i in range(10, 15)]
        >>> features1 = Features()
        >>> features1.local_features = {1: cluster1}
        >>> features2 = Features()
        >>> features2.local_features = {1: cluster2}
        >>>
        >>> clustered_features = selector([features1, features2])
        >>> len(clustered_features)
        1
        >>> result = clustered_features[0]
        >>> sorted(result.local_features.keys())
        [1]
        >>> len(result.local_features[1])
        1
        >>> result.local_features[1][0].shape
        torch.Size([2, 4])
    """

    def __init__(self, num_clusters: int = 3) -> None:
        super().__init__()
        self.num_clusters = num_clusters

    def __call__(self, features_per_image: list[Features]) -> list[Features]:
        """This method clusters all features.

         This is done across all reference images and their masks. This also averages the
         features per cluster.

        Args:
            features_per_image: A list of features for each reference image.

        Returns:
            A list of Features object with the averaged features per cluster.
        """
        result_features = Features()

        for class_id, feature_list in self.get_all_local_class_features(
            features_per_image,
        ).items():
            original_device = feature_list[0].device
            stacked_features = torch.cat(feature_list, dim=0)
            features_np = stacked_features.cpu().numpy()
            kmeans = KMeans(
                n_clusters=self.num_clusters,
                init="k-means++",
                random_state=42,
            )
            kmeans.fit(features_np)

            # use centroid of cluster as prototype
            part_level_features = []
            for c in range(self.num_clusters):
                part_level_feature = features_np[kmeans.labels_ == c].mean(axis=0)
                # Even though input features are normalized, when we take the mean of a cluster's features,
                # the resulting centroid is not guaranteed to have unit norm
                part_level_feature = part_level_feature / np.linalg.norm(
                    part_level_feature,
                    axis=-1,
                    keepdims=True,
                )
                part_level_features.append(torch.from_numpy(part_level_feature))

            part_level_features = torch.stack(part_level_features, dim=0).to(
                original_device,
            )  # n_clusters, embed_dim
            result_features.add_local_features(part_level_features, class_id)

        return [result_features]
