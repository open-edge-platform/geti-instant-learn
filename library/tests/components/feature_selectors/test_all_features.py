# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for AllFeaturesSelector class."""

import pytest
import torch

from getiprompt.components.feature_selectors.all_features import AllFeaturesSelector
from getiprompt.types import Features


class TestAllFeaturesSelector:
    """Test cases for AllFeaturesSelector class."""

    def test_init(self) -> None:
        """Test AllFeaturesSelector initialization."""
        selector = AllFeaturesSelector()
        assert isinstance(selector, AllFeaturesSelector)

    def test_call_empty_list(self) -> None:
        """Test AllFeaturesSelector with empty input list."""
        selector = AllFeaturesSelector()

        # AllFeaturesSelector doesn't handle empty lists - it will raise an error
        with pytest.raises(RuntimeError, match="expected a non-empty list of Tensors"):
            selector([])

    def test_call_single_image(self) -> None:
        """Test AllFeaturesSelector with single image."""
        selector = AllFeaturesSelector()

        features = Features()
        features.global_features = torch.randn(5, 4)
        features.local_features = {
            1: [torch.randn(2, 4), torch.randn(1, 4)],
            2: [torch.randn(3, 4)],
        }

        result = selector([features])

        assert isinstance(result, Features)
        assert result.global_features.shape == (1, 5, 4)
        assert len(result.local_features) == 2
        assert 1 in result.local_features
        assert 2 in result.local_features
        assert len(result.local_features[1]) == 2
        assert len(result.local_features[2]) == 1

    def test_call_multiple_images(self) -> None:
        """Test AllFeaturesSelector with multiple images."""
        selector = AllFeaturesSelector()

        features1 = Features()
        features1.global_features = torch.randn(5, 4)
        features1.local_features = {
            1: [torch.randn(2, 4), torch.randn(1, 4)],
            2: [torch.randn(3, 4)],
        }

        features2 = Features()
        features2.global_features = torch.randn(5, 4)  # Same size as features1
        features2.local_features = {
            1: [torch.randn(1, 4)],
            3: [torch.randn(2, 4), torch.randn(1, 4)],
        }

        result = selector([features1, features2])

        assert isinstance(result, Features)
        assert result.global_features.shape == (2, 5, 4)
        assert len(result.local_features) == 3
        assert 1 in result.local_features
        assert 2 in result.local_features
        assert 3 in result.local_features

    def test_call_preserves_tensor_references(self) -> None:
        """Test that AllFeaturesSelector preserves tensor references."""
        selector = AllFeaturesSelector()

        features = Features()
        features.global_features = torch.randn(5, 4)
        features.local_features = {1: [torch.randn(2, 4)]}

        result = selector([features])

        # Check that tensors are the same objects (not copied)
        assert result.local_features[1][0] is features.local_features[1][0]

    def test_call_different_global_feature_sizes(self) -> None:
        """Test AllFeaturesSelector with different global feature sizes."""
        selector = AllFeaturesSelector()

        features1 = Features()
        features1.global_features = torch.randn(5, 4)

        features2 = Features()
        features2.global_features = torch.randn(3, 4)  # Different size

        # This should raise an error due to size mismatch
        with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
            selector([features1, features2])

    def test_call_no_local_features(self) -> None:
        """Test AllFeaturesSelector with no local features."""
        selector = AllFeaturesSelector()

        features = Features()
        features.global_features = torch.randn(5, 4)
        features.local_features = {}

        result = selector([features])

        assert isinstance(result, Features)
        assert result.global_features.shape == (1, 5, 4)
        assert len(result.local_features) == 0

    def test_call_no_global_features(self) -> None:
        """Test AllFeaturesSelector with no global features."""
        selector = AllFeaturesSelector()

        features = Features()
        features.global_features = torch.randn(0, 4)
        features.local_features = {1: [torch.randn(2, 4)]}

        result = selector([features])

        assert isinstance(result, Features)
        assert result.global_features.shape == (1, 0, 4)
        assert len(result.local_features) == 1

    def test_call_same_class_multiple_images(self) -> None:
        """Test AllFeaturesSelector with same class across multiple images."""
        selector = AllFeaturesSelector()

        features1 = Features()
        features1.global_features = torch.randn(5, 4)
        features1.local_features = {1: [torch.randn(2, 4)]}

        features2 = Features()
        features2.global_features = torch.randn(5, 4)
        features2.local_features = {1: [torch.randn(1, 4)]}

        result = selector([features1, features2])

        assert isinstance(result, Features)
        assert len(result.local_features[1]) == 2  # 1 from features1 + 1 from features2

    def test_call_empty_local_features_list(self) -> None:
        """Test AllFeaturesSelector with empty local features list."""
        selector = AllFeaturesSelector()

        features = Features()
        features.global_features = torch.randn(5, 4)
        features.local_features = {1: []}

        result = selector([features])

        assert isinstance(result, Features)
        assert len(result.local_features[1]) == 0

    def test_call_different_devices(self) -> None:
        """Test AllFeaturesSelector with tensors on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        selector = AllFeaturesSelector()

        features1 = Features()
        features1.global_features = torch.randn(5, 4).cuda()

        features2 = Features()
        features2.global_features = torch.randn(5, 4)  # CPU

        # This should raise an error due to device mismatch
        with pytest.raises(RuntimeError, match="Expected all tensors to be on the same device"):
            selector([features1, features2])

    def test_call_large_batch(self) -> None:
        """Test AllFeaturesSelector with large batch of features."""
        selector = AllFeaturesSelector()

        features_list = []
        for _ in range(10):
            features = Features()
            features.global_features = torch.randn(5, 4)
            features.local_features = {1: [torch.randn(2, 4)]}
            features_list.append(features)

        result = selector(features_list)

        assert isinstance(result, Features)
        assert result.global_features.shape == (10, 5, 4)
        assert len(result.local_features[1]) == 10

    def test_call_returns_single_features_object(self) -> None:
        """Test that AllFeaturesSelector returns a single Features object."""
        selector = AllFeaturesSelector()

        features = Features()
        features.global_features = torch.randn(5, 4)
        features.local_features = {1: [torch.randn(2, 4)]}

        result = selector([features])

        assert isinstance(result, Features)
        assert not isinstance(result, list)
