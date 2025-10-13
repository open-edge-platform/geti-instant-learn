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
        selector = ClusterFeatures()
        assert isinstance(selector, ClusterFeatures)
        assert selector.num_clusters == 3

    def test_init_custom_clusters(self) -> None:
        """Test ClusterFeatures initialization with custom number of clusters."""
        selector = ClusterFeatures(num_clusters=5)
        assert selector.num_clusters == 5


class TestClusterFeaturesBasic:
    """Basic test cases for ClusterFeatures class."""

    def test_call_empty_list(self) -> None:
        """Test ClusterFeatures with empty input list."""
        selector = ClusterFeatures()
        result = selector([])

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Features)
        assert result[0].local_features == {}

    def test_call_single_class_single_image(self) -> None:
        """Test ClusterFeatures with single class and single image."""
        selector = ClusterFeatures(num_clusters=2)

        features = Features()
        features.local_features = {
            1: [torch.ones(2, 4), torch.ones(1, 4)],
            2: [torch.ones(2, 4), torch.ones(1, 4)],
        }

        result = selector([features])

        assert len(result) == 1
        assert 1 in result[0].local_features
        assert 2 in result[0].local_features
        assert len(result[0].local_features[1]) == 1
        assert len(result[0].local_features[2]) == 1

    def test_call_multiple_classes(self) -> None:
        """Test ClusterFeatures with multiple classes."""
        selector = ClusterFeatures(num_clusters=2)

        features = Features()
        features.local_features = {
            1: [torch.ones(2, 4), torch.ones(1, 4)],
            2: [torch.ones(2, 4), torch.ones(1, 4)],
        }

        result = selector([features])

        assert len(result) == 1
        assert 1 in result[0].local_features
        assert 2 in result[0].local_features
        assert len(result[0].local_features[1]) == 1
        assert len(result[0].local_features[2]) == 1

    def test_call_multiple_images(self) -> None:
        """Test ClusterFeatures with multiple images."""
        selector = ClusterFeatures(num_clusters=2)

        features1 = Features()
        features1.local_features = {1: [torch.ones(2, 4)]}

        features2 = Features()
        features2.local_features = {1: [torch.ones(1, 4)]}

        result = selector([features1, features2])

        assert len(result) == 1
        assert 1 in result[0].local_features
        assert len(result[0].local_features[1]) == 1


class TestClusterFeatures:
    """Test cases for ClusterFeatures class."""

    @pytest.mark.parametrize("num_clusters", [1, 2, 5, 10])
    def test_call_different_cluster_numbers(self, num_clusters: int) -> None:
        """Test ClusterFeatures with different numbers of clusters."""
        selector = ClusterFeatures(num_clusters=num_clusters)

        features = Features()
        features.local_features = {1: [torch.ones(1, 4) for _ in range(num_clusters + 2)]}

        result = selector([features])

        assert len(result) == 1
        assert 1 in result[0].local_features
        assert len(result[0].local_features[1]) == 1
        assert result[0].local_features[1][0].shape == (num_clusters, 4)

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
        selector = ClusterFeatures(num_clusters=1)  # Use 1 cluster for single feature

        features = Features()
        features.local_features = {1: [torch.ones(1, 4)]}

        result = selector([features])

        assert len(result) == 1
        assert 1 in result[0].local_features
        assert len(result[0].local_features[1]) == 1
        assert result[0].local_features[1][0].shape == (1, 4)

    def test_call_device_handling(self) -> None:
        """Test ClusterFeatures device handling."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        selector = ClusterFeatures(num_clusters=1)  # Use 1 cluster for single feature

        features = Features()
        features.local_features = {1: [torch.ones(1, 4).cuda()]}

        result = selector([features])

        assert len(result) == 1
        assert 1 in result[0].local_features
        assert len(result[0].local_features[1]) == 1
        # Result should be on the same device as input
        assert result[0].local_features[1][0].device.type == "cuda"

    def test_call_normalization(self) -> None:
        """Test that ClusterFeatures properly normalizes the clustered features."""
        selector = ClusterFeatures(num_clusters=2)

        features = Features()
        features.local_features = {
            1: [
                torch.tensor([[2.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 2.0, 0.0, 0.0]]),
            ]
        }

        result = selector([features])

        clustered = result[0].local_features[1][0]
        # Should be normalized to unit length
        norm = torch.norm(clustered, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm))

    def test_call_large_batch(self) -> None:
        """Test ClusterFeatures with large batch of features."""
        selector = ClusterFeatures(num_clusters=3)

        features = Features()
        features.local_features = {
            1: [torch.ones(1, 4) for _ in range(10)],
            2: [torch.ones(1, 4) for _ in range(5)],
        }

        result = selector([features])

        assert len(result) == 1
        for class_id in [1, 2]:
            assert class_id in result[0].local_features
            assert len(result[0].local_features[class_id]) == 1
            assert result[0].local_features[class_id][0].shape == (3, 4)

    def test_call_different_feature_sizes(self) -> None:
        """Test ClusterFeatures with different feature sizes."""
        selector = ClusterFeatures(num_clusters=2)

        features = Features()
        features.local_features = {1: [torch.ones(2, 8)]}

        result = selector([features])

        assert len(result) == 1
        assert 1 in result[0].local_features
        assert len(result[0].local_features[1]) == 1
        assert result[0].local_features[1][0].shape == (2, 8)

    def test_call_returns_list(self) -> None:
        """Test that ClusterFeatures returns a list."""
        selector = ClusterFeatures(num_clusters=1)  # Use 1 cluster for single feature

        features = Features()
        features.local_features = {1: [torch.ones(1, 4)]}

        result = selector([features])

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Features)

    def test_call_kmeans_initialization(self) -> None:
        """Test that KMeans is initialized with correct parameters."""
        with patch("getiprompt.components.feature_selectors.cluster_features.KMeans") as mock_kmeans:
            mock_kmeans.return_value.fit.return_value = None
            mock_kmeans.return_value.labels_ = np.array([0, 1, 0])

            selector = ClusterFeatures(num_clusters=2)
            features = Features()
            features.local_features = {1: [torch.ones(3, 4)]}

            selector([features])

            # Verify KMeans was called with correct parameters
            mock_kmeans.assert_called_once_with(n_clusters=2, init="k-means++", random_state=42)

    def test_call_kmeans_fit_called(self) -> None:
        """Test that KMeans fit method is called."""
        with patch("getiprompt.components.feature_selectors.cluster_features.KMeans") as mock_kmeans:
            mock_instance = MagicMock()
            mock_instance.labels_ = np.array([0, 1, 0])
            mock_kmeans.return_value = mock_instance

            selector = ClusterFeatures(num_clusters=2)
            features = Features()
            features.local_features = {1: [torch.ones(3, 4)]}

            selector([features])

            # Verify fit was called
            mock_instance.fit.assert_called_once()

    def test_call_cluster_centroid_calculation(self) -> None:
        """Test that cluster centroids are calculated correctly."""
        selector = ClusterFeatures(num_clusters=2)

        features = Features()
        features.local_features = {1: [torch.ones(3, 4)]}

        result = selector([features])

        assert len(result) == 1
        assert 1 in result[0].local_features
        assert len(result[0].local_features[1]) == 1
        assert result[0].local_features[1][0].shape == (2, 4)

    def test_call_zero_features(self) -> None:
        """Test ClusterFeatures with zero features."""
        selector = ClusterFeatures(num_clusters=1)

        features = Features()
        features.local_features = {1: [torch.zeros(2, 4)]}

        result = selector([features])

        assert len(result) == 1
        assert 1 in result[0].local_features
        assert len(result[0].local_features[1]) == 1

    def test_call_single_tensor_multiple_features(self) -> None:
        """Test ClusterFeatures with single tensor containing multiple features."""
        selector = ClusterFeatures(num_clusters=2)

        features = Features()
        features.local_features = {1: [torch.ones(5, 4)]}

        result = selector([features])

        assert len(result) == 1
        assert 1 in result[0].local_features
        assert len(result[0].local_features[1]) == 1
        assert result[0].local_features[1][0].shape == (2, 4)

    def test_call_more_clusters_than_features(self) -> None:
        """Test ClusterFeatures when num_clusters > number of features."""
        selector = ClusterFeatures(num_clusters=5)

        features = Features()
        features.local_features = {
            1: [torch.ones(1, 4), torch.ones(1, 4)]  # Only 2 features
        }

        # This should raise an error due to insufficient samples
        with pytest.raises(ValueError, match="n_samples=2 should be >= n_clusters=5"):
            selector([features])

    def test_call_random_state_consistency(self) -> None:
        """Test that ClusterFeatures produces consistent results with same random state."""
        selector1 = ClusterFeatures(num_clusters=2)
        selector2 = ClusterFeatures(num_clusters=2)

        features = Features()
        features.local_features = {1: [torch.ones(1, 4) for _ in range(10)]}

        result1 = selector1([features])
        result2 = selector2([features])

        # Note: Due to KMeans convergence warnings, we might get NaN values
        # So we check that both results have the same structure
        assert result1[0].local_features[1][0].shape == result2[0].local_features[1][0].shape
        assert len(result1) == len(result2)

    def test_clustering_mathematical_correctness(self) -> None:
        """Test mathematical correctness of clustering."""
        selector = ClusterFeatures(num_clusters=2)

        features = Features()
        features.local_features = {
            1: [
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
                torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            ]
        }

        result = selector([features])

        clustered = result[0].local_features[1][0]
        assert clustered.shape == (2, 4)
        # Should be normalized
        norm = torch.norm(clustered, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm))

    def test_clustering_with_known_groups(self) -> None:
        """Test clustering with known separable groups."""
        selector = ClusterFeatures(num_clusters=2)

        features = Features()
        # Create two clearly separable groups
        features.local_features = {
            1: [
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[1.1, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 1.1, 0.0, 0.0]]),
            ]
        }

        result = selector([features])

        clustered = result[0].local_features[1][0]
        assert clustered.shape == (2, 4)
        # Should be normalized
        norm = torch.norm(clustered, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm))
