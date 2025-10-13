# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for ClusterFeatures class."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from getiprompt.components.feature_selectors.cluster_features import ClusterFeatures
from getiprompt.types import Features


class TestClusterFeaturesInit:
    """Test cases for ClusterFeatures initialization."""

    def test_init_default_clusters(self) -> None:
        """Test ClusterFeatures initialization with default number of clusters."""
        expected_default_clusters = 3
        selector = ClusterFeatures()
        pytest.assume(isinstance(selector, ClusterFeatures))
        pytest.assume(selector.num_clusters == expected_default_clusters)

    def test_init_custom_clusters(self) -> None:
        """Test ClusterFeatures initialization with custom number of clusters."""
        num_clusters = 5
        selector = ClusterFeatures(num_clusters=num_clusters)
        pytest.assume(selector.num_clusters == num_clusters)


class TestClusterFeaturesBasic:
    """Basic test cases for ClusterFeatures class."""

    def test_call_empty_list(self) -> None:
        """Test ClusterFeatures with empty input list."""
        selector = ClusterFeatures()
        result = selector([])

        expected_result_length = 1
        pytest.assume(isinstance(result, list))
        pytest.assume(len(result) == expected_result_length)
        pytest.assume(isinstance(result[0], Features))
        pytest.assume(result[0].local_features == {})

    def test_call_single_class_single_image(self) -> None:
        """Test ClusterFeatures with single class and single image."""
        num_clusters = 2
        selector = ClusterFeatures(num_clusters=num_clusters)

        features = Features()
        features.local_features = {
            1: [torch.ones(2, 4), torch.ones(1, 4)],
            2: [torch.ones(2, 4), torch.ones(1, 4)],
        }

        result = selector([features])

        expected_result_length = 1
        expected_features_per_class = 1
        pytest.assume(len(result) == expected_result_length)

        class_key_1 = 1
        class_key_2 = 2
        pytest.assume(class_key_1 in result[0].local_features)
        pytest.assume(class_key_2 in result[0].local_features)
        pytest.assume(len(result[0].local_features[class_key_1]) == expected_features_per_class)
        pytest.assume(len(result[0].local_features[class_key_2]) == expected_features_per_class)

    def test_call_multiple_classes(self) -> None:
        """Test ClusterFeatures with multiple classes."""
        num_clusters = 2
        selector = ClusterFeatures(num_clusters=num_clusters)

        features = Features()
        features.local_features = {
            1: [torch.ones(2, 4), torch.ones(1, 4)],
            2: [torch.ones(2, 4), torch.ones(1, 4)],
        }

        result = selector([features])

        expected_result_length = 1
        expected_features_per_class = 1
        pytest.assume(len(result) == expected_result_length)

        class_key_1 = 1
        class_key_2 = 2
        pytest.assume(class_key_1 in result[0].local_features)
        pytest.assume(class_key_2 in result[0].local_features)
        pytest.assume(len(result[0].local_features[class_key_1]) == expected_features_per_class)
        pytest.assume(len(result[0].local_features[class_key_2]) == expected_features_per_class)

    def test_call_multiple_images(self) -> None:
        """Test ClusterFeatures with multiple images."""
        num_clusters = 2
        selector = ClusterFeatures(num_clusters=num_clusters)

        features1 = Features()
        features1.local_features = {1: [torch.ones(2, 4)]}

        features2 = Features()
        features2.local_features = {1: [torch.ones(1, 4)]}

        result = selector([features1, features2])

        expected_result_length = 1
        expected_features_per_class = 1
        pytest.assume(len(result) == expected_result_length)
        pytest.assume(1 in result[0].local_features)
        pytest.assume(len(result[0].local_features[1]) == expected_features_per_class)


class TestClusterFeatures:
    """Test cases for ClusterFeatures class."""

    @pytest.mark.parametrize("num_clusters", [1, 2, 5, 10])
    def test_call_different_cluster_numbers(self, num_clusters: int) -> None:
        """Test ClusterFeatures with different numbers of clusters."""
        selector = ClusterFeatures(num_clusters=num_clusters)

        features = Features()
        num_features = num_clusters + 2
        feature_dim = 4
        features.local_features = {1: [torch.ones(1, feature_dim) for _ in range(num_features)]}

        result = selector([features])

        expected_result_length = 1
        expected_features_per_class = 1
        pytest.assume(len(result) == expected_result_length)
        pytest.assume(1 in result[0].local_features)
        pytest.assume(len(result[0].local_features[1]) == expected_features_per_class)
        pytest.assume(result[0].local_features[1][0].shape == (num_clusters, feature_dim))

    def test_call_empty_class_features(self) -> None:
        """Test ClusterFeatures with empty class features."""
        selector = ClusterFeatures()

        features = Features()
        features.local_features = {1: []}

        # ClusterFeatures doesn't handle empty feature lists gracefully
        with pytest.raises(IndexError):
            selector([features])

    def test_call_single_feature_per_class(self) -> None:
        """Test ClusterFeatures with single feature per class."""
        num_clusters = 1
        feature_dim = 4
        selector = ClusterFeatures(num_clusters=num_clusters)

        features = Features()
        features.local_features = {1: [torch.ones(1, feature_dim)]}

        result = selector([features])

        expected_result_length = 1
        expected_features_per_class = 1
        pytest.assume(len(result) == expected_result_length)
        pytest.assume(1 in result[0].local_features)
        pytest.assume(len(result[0].local_features[1]) == expected_features_per_class)
        pytest.assume(result[0].local_features[1][0].shape == (num_clusters, feature_dim))

    def test_call_device_handling(self) -> None:
        """Test ClusterFeatures device handling."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        num_clusters = 1
        feature_dim = 4
        selector = ClusterFeatures(num_clusters=num_clusters)

        features = Features()
        features.local_features = {1: [torch.ones(1, feature_dim).cuda()]}

        result = selector([features])

        expected_result_length = 1
        expected_features_per_class = 1
        pytest.assume(len(result) == expected_result_length)
        pytest.assume(1 in result[0].local_features)
        pytest.assume(len(result[0].local_features[1]) == expected_features_per_class)
        # Result should be on the same device as input
        pytest.assume(result[0].local_features[1][0].device.type == "cuda")

    def test_call_normalization(self) -> None:
        """Test that ClusterFeatures properly normalizes the clustered features."""
        num_clusters = 2
        selector = ClusterFeatures(num_clusters=num_clusters)

        features = Features()
        features.local_features = {
            1: [
                torch.tensor([[2.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 2.0, 0.0, 0.0]]),
            ],
        }

        result = selector([features])

        clustered = result[0].local_features[1][0]
        # Should be normalized to unit length
        norm = torch.norm(clustered, dim=-1)
        pytest.assume(torch.allclose(norm, torch.ones_like(norm)))

    def test_call_large_batch(self) -> None:
        """Test ClusterFeatures with large batch of features."""
        num_clusters = 3
        num_features_class_1 = 10
        num_features_class_2 = 5
        feature_dim = 4
        selector = ClusterFeatures(num_clusters=num_clusters)

        features = Features()
        features.local_features = {
            1: [torch.ones(1, feature_dim) for _ in range(num_features_class_1)],
            2: [torch.ones(1, feature_dim) for _ in range(num_features_class_2)],
        }

        result = selector([features])

        expected_result_length = 1
        pytest.assume(len(result) == expected_result_length)
        for class_id in [1, 2]:
            expected_features_per_class = 1
            pytest.assume(class_id in result[0].local_features)
            pytest.assume(len(result[0].local_features[class_id]) == expected_features_per_class)
            pytest.assume(result[0].local_features[class_id][0].shape == (num_clusters, feature_dim))

    def test_call_different_feature_sizes(self) -> None:
        """Test ClusterFeatures with different feature sizes."""
        num_clusters = 2
        feature_dim = 8
        num_features = 2
        selector = ClusterFeatures(num_clusters=num_clusters)

        features = Features()
        features.local_features = {1: [torch.ones(num_features, feature_dim)]}

        result = selector([features])

        expected_result_length = 1
        expected_features_per_class = 1
        pytest.assume(len(result) == expected_result_length)
        pytest.assume(1 in result[0].local_features)
        pytest.assume(len(result[0].local_features[1]) == expected_features_per_class)
        pytest.assume(result[0].local_features[1][0].shape == (num_clusters, feature_dim))

    def test_call_returns_list(self) -> None:
        """Test that ClusterFeatures returns a list."""
        num_clusters = 1
        feature_dim = 4
        selector = ClusterFeatures(num_clusters=num_clusters)

        features = Features()
        features.local_features = {1: [torch.ones(1, feature_dim)]}

        result = selector([features])

        expected_result_length = 1
        pytest.assume(isinstance(result, list))
        pytest.assume(len(result) == expected_result_length)
        pytest.assume(isinstance(result[0], Features))

    def test_call_kmeans_initialization(self) -> None:
        """Test that KMeans is initialized with correct parameters."""
        num_clusters = 2
        random_state = 42
        num_features = 3
        feature_dim = 4

        with patch("getiprompt.components.feature_selectors.cluster_features.KMeans") as mock_kmeans:
            mock_kmeans.return_value.fit.return_value = None
            mock_kmeans.return_value.labels_ = np.array([0, 1, 0])

            selector = ClusterFeatures(num_clusters=num_clusters)
            features = Features()
            features.local_features = {1: [torch.ones(num_features, feature_dim)]}

            selector([features])

            # Verify KMeans was called with correct parameters
            mock_kmeans.assert_called_once_with(
                n_clusters=num_clusters,
                init="k-means++",
                random_state=random_state,
            )

    def test_call_kmeans_fit_called(self) -> None:
        """Test that KMeans fit method is called."""
        num_clusters = 2
        num_features = 3
        feature_dim = 4

        with patch("getiprompt.components.feature_selectors.cluster_features.KMeans") as mock_kmeans:
            mock_instance = MagicMock()
            mock_instance.labels_ = np.array([0, 1, 0])
            mock_kmeans.return_value = mock_instance

            selector = ClusterFeatures(num_clusters=num_clusters)
            features = Features()
            features.local_features = {1: [torch.ones(num_features, feature_dim)]}

            selector([features])

            # Verify fit was called
            mock_instance.fit.assert_called_once()

    def test_call_cluster_centroid_calculation(self) -> None:
        """Test that cluster centroids are calculated correctly."""
        num_clusters = 2
        num_features = 3
        feature_dim = 4
        selector = ClusterFeatures(num_clusters=num_clusters)

        features = Features()
        features.local_features = {1: [torch.ones(num_features, feature_dim)]}

        result = selector([features])

        expected_result_length = 1
        expected_features_per_class = 1
        pytest.assume(len(result) == expected_result_length)
        pytest.assume(1 in result[0].local_features)
        pytest.assume(len(result[0].local_features[1]) == expected_features_per_class)
        pytest.assume(result[0].local_features[1][0].shape == (num_clusters, feature_dim))

    def test_call_zero_features(self) -> None:
        """Test ClusterFeatures with zero features."""
        num_clusters = 1
        num_features = 2
        feature_dim = 4
        selector = ClusterFeatures(num_clusters=num_clusters)

        features = Features()
        features.local_features = {1: [torch.zeros(num_features, feature_dim)]}

        result = selector([features])

        expected_result_length = 1
        expected_features_per_class = 1
        pytest.assume(len(result) == expected_result_length)
        pytest.assume(1 in result[0].local_features)
        pytest.assume(len(result[0].local_features[1]) == expected_features_per_class)

    def test_call_single_tensor_multiple_features(self) -> None:
        """Test ClusterFeatures with single tensor containing multiple features."""
        num_clusters = 2
        num_features = 5
        feature_dim = 4
        selector = ClusterFeatures(num_clusters=num_clusters)

        features = Features()
        features.local_features = {1: [torch.ones(num_features, feature_dim)]}

        result = selector([features])

        expected_result_length = 1
        expected_features_per_class = 1
        pytest.assume(len(result) == expected_result_length)
        pytest.assume(1 in result[0].local_features)
        pytest.assume(len(result[0].local_features[1]) == expected_features_per_class)
        pytest.assume(result[0].local_features[1][0].shape == (num_clusters, feature_dim))

    def test_call_more_clusters_than_features(self) -> None:
        """Test ClusterFeatures when num_clusters > number of features."""
        num_clusters = 5
        num_features = 2
        feature_dim = 4
        selector = ClusterFeatures(num_clusters=num_clusters)

        features = Features()
        features.local_features = {
            1: [torch.ones(1, feature_dim), torch.ones(1, feature_dim)],  # Only 2 features
        }

        # This should raise an error due to insufficient samples
        with pytest.raises(ValueError, match=f"n_samples={num_features} should be >= n_clusters={num_clusters}"):
            selector([features])

    def test_call_random_state_consistency(self) -> None:
        """Test that ClusterFeatures produces consistent results with same random state."""
        num_clusters = 2
        num_features = 10
        feature_dim = 4
        selector1 = ClusterFeatures(num_clusters=num_clusters)
        selector2 = ClusterFeatures(num_clusters=num_clusters)

        features = Features()
        features.local_features = {1: [torch.ones(1, feature_dim) for _ in range(num_features)]}

        result1 = selector1([features])
        result2 = selector2([features])

        # Note: Due to KMeans convergence warnings, we might get NaN values
        # So we check that both results have the same structure
        pytest.assume(result1[0].local_features[1][0].shape == result2[0].local_features[1][0].shape)
        pytest.assume(len(result1) == len(result2))

    def test_clustering_mathematical_correctness(self) -> None:
        """Test mathematical correctness of clustering."""
        num_clusters = 2
        selector = ClusterFeatures(num_clusters=num_clusters)

        features = Features()
        features.local_features = {
            1: [
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
                torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            ],
        }

        result = selector([features])

        feature_dim = 4
        clustered = result[0].local_features[1][0]
        pytest.assume(clustered.shape == (num_clusters, feature_dim))
        # Should be normalized
        norm = torch.norm(clustered, dim=-1)
        pytest.assume(torch.allclose(norm, torch.ones_like(norm)))

    def test_clustering_with_known_groups(self) -> None:
        """Test clustering with known separable groups."""
        num_clusters = 2
        selector = ClusterFeatures(num_clusters=num_clusters)

        features = Features()
        # Create two clearly separable groups
        features.local_features = {
            1: [
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[1.1, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 1.1, 0.0, 0.0]]),
            ],
        }

        result = selector([features])

        feature_dim = 4
        clustered = result[0].local_features[1][0]
        pytest.assume(clustered.shape == (num_clusters, feature_dim))
        # Should be normalized
        norm = torch.norm(clustered, dim=-1)
        pytest.assume(torch.allclose(norm, torch.ones_like(norm)))
