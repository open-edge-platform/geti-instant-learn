# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for FeatureSelector base class."""

import pytest
import torch

from getiprompt.components.feature_selectors.all_features import AllFeaturesSelector
from getiprompt.components.feature_selectors.average_features import AverageFeatures
from getiprompt.components.feature_selectors.base import FeatureSelector
from getiprompt.types import Features


class TestFeatureSelector:
    """Test cases for FeatureSelector base class."""

    def test_abstract_method_raises_error(self) -> None:
        """Test that calling abstract method raises TypeError on instantiation."""
        # FeatureSelector is abstract and cannot be instantiated
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            FeatureSelector()

    def test_get_all_local_class_features_empty_list(self) -> None:
        """Test get_all_local_class_features with empty list."""
        result = FeatureSelector.get_all_local_class_features([])

        pytest.assume(isinstance(result, dict))
        expected_length = 0
        pytest.assume(len(result) == expected_length)

    def test_get_all_local_class_features_single_image(self) -> None:
        """Test get_all_local_class_features with single image."""
        features = Features()
        features.local_features = {
            1: [torch.randn(2, 4), torch.randn(1, 4)],
            2: [torch.randn(3, 4)],
        }

        result = FeatureSelector.get_all_local_class_features([features])

        pytest.assume(isinstance(result, dict))
        expected_num_classes = 2
        expected_class_1_features = 2
        expected_class_2_features = 1
        pytest.assume(len(result) == expected_num_classes)
        class_key_1 = 1
        class_key_2 = 2
        pytest.assume(class_key_1 in result)
        pytest.assume(class_key_2 in result)
        pytest.assume(len(result[class_key_1]) == expected_class_1_features)
        pytest.assume(len(result[class_key_2]) == expected_class_2_features)

    def test_get_all_local_class_features_multiple_images(self) -> None:
        """Test get_all_local_class_features with multiple images."""
        features1 = Features()
        features1.local_features = {
            1: [torch.randn(2, 4), torch.randn(1, 4)],
            2: [torch.randn(3, 4)],
        }

        features2 = Features()
        features2.local_features = {
            1: [torch.randn(1, 4)],
            3: [torch.randn(2, 4), torch.randn(1, 4)],
        }

        result = FeatureSelector.get_all_local_class_features([features1, features2])

        pytest.assume(isinstance(result, dict))
        expected_num_classes = 3
        expected_class_1_features = 3  # 2 + 1
        expected_class_2_features = 1
        expected_class_3_features = 2
        pytest.assume(len(result) == expected_num_classes)
        class_key_1 = 1
        class_key_2 = 2
        class_key_3 = 3
        pytest.assume(class_key_1 in result)
        pytest.assume(class_key_2 in result)
        pytest.assume(class_key_3 in result)
        pytest.assume(len(result[class_key_1]) == expected_class_1_features)
        pytest.assume(len(result[class_key_2]) == expected_class_2_features)
        pytest.assume(len(result[class_key_3]) == expected_class_3_features)

    def test_get_all_local_class_features_overlapping_classes(self) -> None:
        """Test get_all_local_class_features with overlapping classes."""
        features1 = Features()
        features1.local_features = {
            1: [torch.randn(1, 4)],
            2: [torch.randn(1, 4)],
        }

        features2 = Features()
        features2.local_features = {
            1: [torch.randn(1, 4), torch.randn(1, 4)],
            2: [torch.randn(1, 4)],
        }

        result = FeatureSelector.get_all_local_class_features([features1, features2])

        pytest.assume(isinstance(result, dict))
        expected_num_classes = 2
        expected_class_1_features = 3  # 1 + 2
        expected_class_2_features = 2  # 1 + 1
        pytest.assume(len(result) == expected_num_classes)
        class_key_1 = 1
        class_key_2 = 2
        pytest.assume(class_key_1 in result)
        pytest.assume(class_key_2 in result)
        pytest.assume(len(result[class_key_1]) == expected_class_1_features)
        pytest.assume(len(result[class_key_2]) == expected_class_2_features)

    def test_get_all_local_class_features_preserves_tensors(self) -> None:
        """Test that get_all_local_class_features preserves tensor references."""
        features = Features()
        tensor1 = torch.randn(2, 4)
        tensor2 = torch.randn(1, 4)
        features.local_features = {1: [tensor1, tensor2]}

        result = FeatureSelector.get_all_local_class_features([features])

        # Check that tensors are the same objects (not copied)
        pytest.assume(result[1][0] is tensor1)
        pytest.assume(result[1][1] is tensor2)

    def test_get_all_local_class_features_different_devices(self) -> None:
        """Test get_all_local_class_features with tensors on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        features1 = Features()
        features1.local_features = {1: [torch.randn(2, 4).cuda()]}

        features2 = Features()
        features2.local_features = {1: [torch.randn(1, 4)]}  # CPU

        result = FeatureSelector.get_all_local_class_features([features1, features2])

        pytest.assume(isinstance(result, dict))
        expected_num_features = 2
        pytest.assume(1 in result)
        pytest.assume(len(result[1]) == expected_num_features)

    def test_inheritance_works_correctly(self) -> None:
        """Test that concrete classes inherit from FeatureSelector correctly."""
        pytest.assume(issubclass(AllFeaturesSelector, FeatureSelector))
        pytest.assume(issubclass(AverageFeatures, FeatureSelector))

    def test_get_all_local_class_features_with_none_features(self) -> None:
        """Test get_all_local_class_features with None features."""
        features = Features()
        features.local_features = None

        # The implementation doesn't handle None local_features gracefully
        with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'items'"):
            FeatureSelector.get_all_local_class_features([features])

    def test_get_all_local_class_features_with_empty_local_features(self) -> None:
        """Test get_all_local_class_features with empty local_features."""
        features = Features()
        features.local_features = {}

        result = FeatureSelector.get_all_local_class_features([features])

        pytest.assume(isinstance(result, dict))
        expected_length = 0
        pytest.assume(len(result) == expected_length)

    def test_get_all_local_class_features_with_empty_feature_lists(self) -> None:
        """Test get_all_local_class_features with empty feature lists."""
        features = Features()
        features.local_features = {1: [], 2: []}

        result = FeatureSelector.get_all_local_class_features([features])

        pytest.assume(isinstance(result, dict))
        expected_num_classes = 2
        expected_empty_list_length = 0
        class_key_1 = 1
        class_key_2 = 2
        pytest.assume(len(result) == expected_num_classes)
        pytest.assume(class_key_1 in result)
        pytest.assume(class_key_2 in result)
        pytest.assume(len(result[class_key_1]) == expected_empty_list_length)
        pytest.assume(len(result[class_key_2]) == expected_empty_list_length)
