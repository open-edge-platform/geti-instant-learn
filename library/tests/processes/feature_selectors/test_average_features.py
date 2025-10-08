# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for AverageFeatures class."""

import pytest
import torch

from getiprompt.processes.feature_selectors.average_features import AverageFeatures
from getiprompt.types import Features


class TestAverageFeatures:
    """Test cases for AverageFeatures class."""

    def test_init(self) -> None:
        """Test AverageFeatures initialization."""
        selector = AverageFeatures()
        assert isinstance(selector, AverageFeatures)

    def test_call_empty_list(self) -> None:
        """Test AverageFeatures with empty input list."""
        selector = AverageFeatures()
        result = selector([])

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Features)
        assert result[0].local_features == {}

    def test_call_single_image_single_class(self) -> None:
        """Test AverageFeatures with single image and single class."""
        selector = AverageFeatures()

        features = Features()
        features.local_features = {1: [torch.ones(2, 4), torch.ones(1, 4)]}

        result = selector([features])

        assert len(result) == 1
        assert 1 in result[0].local_features
        assert len(result[0].local_features[1]) == 1
        assert result[0].local_features[1][0].shape == (1, 4)

    def test_call_multiple_images_single_class(self) -> None:
        """Test AverageFeatures with multiple images and single class."""
        selector = AverageFeatures()

        features1 = Features()
        features1.local_features = {1: [torch.ones(2, 4)]}

        features2 = Features()
        features2.local_features = {1: [torch.ones(1, 4)]}

        result = selector([features1, features2])

        assert len(result) == 1
        assert 1 in result[0].local_features
        assert len(result[0].local_features[1]) == 1
        assert result[0].local_features[1][0].shape == (1, 4)

    def test_call_multiple_classes(self) -> None:
        """Test AverageFeatures with multiple classes."""
        selector = AverageFeatures()

        features = Features()
        features.local_features = {
            1: [torch.ones(2, 4), torch.ones(1, 4)],
            2: [torch.ones(3, 4)],
        }

        result = selector([features])

        assert len(result) == 1
        for class_id in [1, 2]:
            assert class_id in result[0].local_features
            assert len(result[0].local_features[class_id]) == 1
            assert result[0].local_features[class_id][0].shape == (1, 4)

    def test_call_normalization(self) -> None:
        """Test that AverageFeatures properly normalizes the averaged features."""
        selector = AverageFeatures()

        features = Features()
        features.local_features = {
            1: [
                torch.tensor([[2.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 2.0, 0.0, 0.0]]),
            ]
        }

        result = selector([features])

        averaged = result[0].local_features[1][0]
        # Should be normalized to unit length
        norm = torch.norm(averaged, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm))

    @pytest.mark.parametrize("feature_dims", [4, 8, 16, 32])
    def test_call_different_feature_sizes(self, feature_dims: int) -> None:
        """Test AverageFeatures with different feature sizes."""
        selector = AverageFeatures()

        features = Features()
        features.local_features = {
            1: [
                torch.ones(1, feature_dims),
                torch.ones(2, feature_dims),
                torch.ones(3, feature_dims),
            ]
        }

        result = selector([features])

        assert len(result) == 1
        assert 1 in result[0].local_features
        assert len(result[0].local_features[1]) == 1
        assert result[0].local_features[1][0].shape == (1, feature_dims)

    def test_call_empty_class_features(self) -> None:
        """Test AverageFeatures with empty class features."""
        selector = AverageFeatures()

        features = Features()
        features.local_features = {1: []}

        # AverageFeatures doesn't handle empty feature lists gracefully
        with pytest.raises(RuntimeError, match="expected a non-empty list of Tensors"):
            selector([features])

    def test_call_single_feature_per_class(self) -> None:
        """Test AverageFeatures with single feature per class."""
        selector = AverageFeatures()

        features = Features()
        features.local_features = {
            1: [torch.ones(1, 4)],
            2: [torch.ones(1, 4)],
            3: [torch.ones(1, 4)],
        }

        result = selector([features])

        assert len(result) == 1
        for class_id in [1, 2, 3]:
            assert class_id in result[0].local_features
            assert len(result[0].local_features[class_id]) == 1
            assert result[0].local_features[class_id][0].shape == (1, 4)

    def test_call_multiple_images_multiple_classes(self) -> None:
        """Test AverageFeatures with multiple images and multiple classes."""
        selector = AverageFeatures()

        features1 = Features()
        features1.local_features = {
            1: [torch.ones(2, 4)],
            2: [torch.ones(1, 4)],
        }

        features2 = Features()
        features2.local_features = {
            1: [torch.ones(1, 4)],
            3: [torch.ones(2, 4)],
        }

        result = selector([features1, features2])

        assert len(result) == 1
        assert 1 in result[0].local_features
        assert 2 in result[0].local_features
        assert 3 in result[0].local_features
        assert len(result[0].local_features[1]) == 1
        assert len(result[0].local_features[2]) == 1
        assert len(result[0].local_features[3]) == 1

    def test_call_different_devices(self) -> None:
        """Test AverageFeatures with tensors on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        selector = AverageFeatures()

        features1 = Features()
        features1.local_features = {1: [torch.ones(2, 4).cuda()]}

        features2 = Features()
        features2.local_features = {1: [torch.ones(1, 4)]}  # CPU

        # AverageFeatures requires tensors on same device
        with pytest.raises(RuntimeError, match="Expected all tensors to be on the same device"):
            selector([features1, features2])

    def test_call_large_batch(self) -> None:
        """Test AverageFeatures with large batch of features."""
        selector = AverageFeatures()

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
            assert result[0].local_features[class_id][0].shape == (1, 4)

    def test_call_zero_features(self) -> None:
        """Test AverageFeatures with zero features."""
        selector = AverageFeatures()

        features = Features()
        features.local_features = {1: [torch.zeros(2, 4)]}

        result = selector([features])

        averaged = result[0].local_features[1][0]
        # Zero features result in NaN after normalization
        assert torch.isnan(averaged).all()

    def test_call_returns_list(self) -> None:
        """Test that AverageFeatures returns a list."""
        selector = AverageFeatures()

        features = Features()
        features.local_features = {1: [torch.ones(1, 4)]}

        result = selector([features])

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Features)

    def test_call_preserves_feature_dimensions(self) -> None:
        """Test that AverageFeatures preserves feature dimensions."""
        selector = AverageFeatures()

        features = Features()
        features.local_features = {1: [torch.ones(2, 8)]}

        result = selector([features])

        assert result[0].local_features[1][0].shape == (1, 8)

    def test_call_handles_single_tensor_per_class(self) -> None:
        """Test AverageFeatures with single tensor per class."""
        selector = AverageFeatures()

        features = Features()
        features.local_features = {1: [torch.ones(3, 4)]}

        result = selector([features])

        assert len(result) == 1
        assert 1 in result[0].local_features
        assert len(result[0].local_features[1]) == 1
        assert result[0].local_features[1][0].shape == (1, 4)

    def test_averaging_mathematical_correctness(self) -> None:
        """Test mathematical correctness of averaging."""
        selector = AverageFeatures()

        features = Features()
        features.local_features = {
            1: [
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            ]
        }

        result = selector([features])

        averaged = result[0].local_features[1][0]
        expected = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        expected_normalized = expected / torch.norm(expected, dim=-1, keepdim=True)

        assert torch.allclose(averaged, expected_normalized, atol=1e-6)

    def test_averaging_with_known_weights(self) -> None:
        """Test averaging with known feature weights."""
        selector = AverageFeatures()

        features = Features()
        features.local_features = {
            1: [
                torch.tensor([[2.0, 0.0, 0.0, 0.0]]),  # 1 feature
                torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),  # 2 features
            ]
        }

        result = selector([features])

        averaged = result[0].local_features[1][0]
        # Should be normalized
        norm = torch.norm(averaged, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm))
