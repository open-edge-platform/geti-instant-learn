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
        pytest.assume(isinstance(selector, AllFeaturesSelector))

    def test_call_empty_list(self) -> None:
        """Test AllFeaturesSelector with empty input list."""
        selector = AllFeaturesSelector()

        # AllFeaturesSelector doesn't handle empty lists - it will raise an error
        with pytest.raises(RuntimeError, match="expected a non-empty list of Tensors"):
            selector([])

    def test_call_single_image(self) -> None:
        """Test AllFeaturesSelector with single image."""
        selector = AllFeaturesSelector()

        num_global_features = 5
        feature_dim = 4
        num_class_1_feature_tensors = 2
        num_class_2_feature_tensors = 1
        class_1_tensor_1_size = 2
        class_1_tensor_2_size = 1
        class_2_tensor_size = 3

        features = Features()
        features.global_features = torch.randn(num_global_features, feature_dim)
        features.local_features = {
            1: [torch.randn(class_1_tensor_1_size, feature_dim), torch.randn(class_1_tensor_2_size, feature_dim)],
            2: [torch.randn(class_2_tensor_size, feature_dim)],
        }

        result = selector([features])

        expected_batch_size = 1
        expected_num_local_classes = 2
        expected_global_shape = (expected_batch_size, num_global_features, feature_dim)
        pytest.assume(isinstance(result, Features))
        pytest.assume(result.global_features.shape == expected_global_shape)
        pytest.assume(len(result.local_features) == expected_num_local_classes)

        class_keys = [1, 2]
        for class_id in class_keys:
            pytest.assume(class_id in result.local_features)
        pytest.assume(len(result.local_features[1]) == num_class_1_feature_tensors)
        pytest.assume(len(result.local_features[2]) == num_class_2_feature_tensors)

    def test_call_multiple_images(self) -> None:
        """Test AllFeaturesSelector with multiple images."""
        selector = AllFeaturesSelector()

        num_global_features = 5
        feature_dim = 4
        class_1_tensor_1_size = 2
        class_1_tensor_2_size = 1
        class_2_tensor_size = 3
        class_3_tensor_1_size = 2
        class_3_tensor_2_size = 1

        features1 = Features()
        features1.global_features = torch.randn(num_global_features, feature_dim)
        features1.local_features = {
            1: [torch.randn(class_1_tensor_1_size, feature_dim), torch.randn(class_1_tensor_2_size, feature_dim)],
            2: [torch.randn(class_2_tensor_size, feature_dim)],
        }

        features2 = Features()
        features2.global_features = torch.randn(num_global_features, feature_dim)  # Same size as features1
        features2.local_features = {
            1: [torch.randn(class_1_tensor_2_size, feature_dim)],
            3: [torch.randn(class_3_tensor_1_size, feature_dim), torch.randn(class_3_tensor_2_size, feature_dim)],
        }

        result = selector([features1, features2])

        expected_batch_size = 2
        expected_num_local_classes = 3
        expected_global_shape = (expected_batch_size, num_global_features, feature_dim)
        pytest.assume(isinstance(result, Features))
        pytest.assume(result.global_features.shape == expected_global_shape)
        pytest.assume(len(result.local_features) == expected_num_local_classes)
        class_keys = [1, 2, 3]
        for class_id in class_keys:
            pytest.assume(class_id in result.local_features)

    def test_call_preserves_tensor_references(self) -> None:
        """Test that AllFeaturesSelector preserves tensor references."""
        selector = AllFeaturesSelector()

        num_global_features = 5
        feature_dim = 4
        num_features = 2

        features = Features()
        features.global_features = torch.randn(num_global_features, feature_dim)
        features.local_features = {1: [torch.randn(num_features, feature_dim)]}

        result = selector([features])

        # Check that tensors are the same objects (not copied)
        pytest.assume(result.local_features[1][0] is features.local_features[1][0])

    def test_call_different_global_feature_sizes(self) -> None:
        """Test AllFeaturesSelector with different global feature sizes."""
        selector = AllFeaturesSelector()

        num_global_features_1 = 5
        num_global_features_2 = 3
        feature_dim = 4

        features1 = Features()
        features1.global_features = torch.randn(num_global_features_1, feature_dim)

        features2 = Features()
        features2.global_features = torch.randn(num_global_features_2, feature_dim)  # Different size

        # This should raise an error due to size mismatch
        with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
            selector([features1, features2])

    def test_call_no_local_features(self) -> None:
        """Test AllFeaturesSelector with no local features."""
        selector = AllFeaturesSelector()

        num_global_features = 5
        feature_dim = 4

        features = Features()
        features.global_features = torch.randn(num_global_features, feature_dim)
        features.local_features = {}

        result = selector([features])

        expected_batch_size = 1
        expected_num_local_classes = 0
        expected_global_shape = (expected_batch_size, num_global_features, feature_dim)
        pytest.assume(isinstance(result, Features))
        pytest.assume(result.global_features.shape == expected_global_shape)
        pytest.assume(len(result.local_features) == expected_num_local_classes)

    def test_call_no_global_features(self) -> None:
        """Test AllFeaturesSelector with no global features."""
        selector = AllFeaturesSelector()

        num_global_features = 0
        feature_dim = 4
        num_local_features = 2

        features = Features()
        features.global_features = torch.randn(num_global_features, feature_dim)
        features.local_features = {1: [torch.randn(num_local_features, feature_dim)]}

        result = selector([features])

        expected_batch_size = 1
        expected_num_local_classes = 1
        expected_global_shape = (expected_batch_size, num_global_features, feature_dim)
        pytest.assume(isinstance(result, Features))
        pytest.assume(result.global_features.shape == expected_global_shape)
        pytest.assume(len(result.local_features) == expected_num_local_classes)

    def test_call_same_class_multiple_images(self) -> None:
        """Test AllFeaturesSelector with same class across multiple images."""
        selector = AllFeaturesSelector()

        num_global_features = 5
        feature_dim = 4
        num_features_image_1 = 2
        num_features_image_2 = 1

        features1 = Features()
        features1.global_features = torch.randn(num_global_features, feature_dim)
        features1.local_features = {1: [torch.randn(num_features_image_1, feature_dim)]}

        features2 = Features()
        features2.global_features = torch.randn(num_global_features, feature_dim)
        features2.local_features = {1: [torch.randn(num_features_image_2, feature_dim)]}

        result = selector([features1, features2])

        expected_class_1_features = 2  # 1 from features1 + 1 from features2
        pytest.assume(isinstance(result, Features))
        pytest.assume(len(result.local_features[1]) == expected_class_1_features)

    def test_call_empty_local_features_list(self) -> None:
        """Test AllFeaturesSelector with empty local features list."""
        selector = AllFeaturesSelector()

        num_global_features = 5
        feature_dim = 4

        features = Features()
        features.global_features = torch.randn(num_global_features, feature_dim)
        features.local_features = {1: []}

        result = selector([features])

        expected_num_features = 0
        pytest.assume(isinstance(result, Features))
        pytest.assume(len(result.local_features[1]) == expected_num_features)

    def test_call_different_devices(self) -> None:
        """Test AllFeaturesSelector with tensors on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        selector = AllFeaturesSelector()

        num_global_features = 5
        feature_dim = 4

        features1 = Features()
        features1.global_features = torch.randn(num_global_features, feature_dim).cuda()

        features2 = Features()
        features2.global_features = torch.randn(num_global_features, feature_dim)  # CPU

        # This should raise an error due to device mismatch
        with pytest.raises(RuntimeError, match="Expected all tensors to be on the same device"):
            selector([features1, features2])

    def test_call_large_batch(self) -> None:
        """Test AllFeaturesSelector with large batch of features."""
        selector = AllFeaturesSelector()

        batch_size = 10
        num_global_features = 5
        feature_dim = 4
        num_local_features = 2

        features_list = []
        for _ in range(batch_size):
            features = Features()
            features.global_features = torch.randn(num_global_features, feature_dim)
            features.local_features = {1: [torch.randn(num_local_features, feature_dim)]}
            features_list.append(features)

        result = selector(features_list)

        expected_global_shape = (batch_size, num_global_features, feature_dim)
        expected_num_features = batch_size
        pytest.assume(isinstance(result, Features))
        pytest.assume(result.global_features.shape == expected_global_shape)
        pytest.assume(len(result.local_features[1]) == expected_num_features)

    def test_call_returns_single_features_object(self) -> None:
        """Test that AllFeaturesSelector returns a single Features object."""
        selector = AllFeaturesSelector()

        num_global_features = 5
        feature_dim = 4
        num_local_features = 2

        features = Features()
        features.global_features = torch.randn(num_global_features, feature_dim)
        features.local_features = {1: [torch.randn(num_local_features, feature_dim)]}

        result = selector([features])

        pytest.assume(isinstance(result, Features))
        pytest.assume(not isinstance(result, list))
