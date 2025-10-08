# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for FeatureSelector base class."""

import pytest
import torch

from getiprompt.processes.feature_selectors.feature_selector_base import FeatureSelector
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

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_get_all_local_class_features_single_image(self) -> None:
        """Test get_all_local_class_features with single image."""
        features = Features()
        features.local_features = {
            1: [torch.randn(2, 4), torch.randn(1, 4)],
            2: [torch.randn(3, 4)],
        }

        result = FeatureSelector.get_all_local_class_features([features])

        assert isinstance(result, dict)
        assert len(result) == 2
        assert 1 in result
        assert 2 in result
        assert len(result[1]) == 2
        assert len(result[2]) == 1

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

        assert isinstance(result, dict)
        assert len(result) == 3
        assert 1 in result
        assert 2 in result
        assert 3 in result
        assert len(result[1]) == 3  # 2 + 1
        assert len(result[2]) == 1
        assert len(result[3]) == 2  # 2 + 1

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

        assert isinstance(result, dict)
        assert len(result) == 2
        assert 1 in result
        assert 2 in result
        assert len(result[1]) == 3  # 1 + 2
        assert len(result[2]) == 2  # 1 + 1

    def test_get_all_local_class_features_preserves_tensors(self) -> None:
        """Test that get_all_local_class_features preserves tensor references."""
        features = Features()
        tensor1 = torch.randn(2, 4)
        tensor2 = torch.randn(1, 4)
        features.local_features = {1: [tensor1, tensor2]}

        result = FeatureSelector.get_all_local_class_features([features])

        # Check that tensors are the same objects (not copied)
        assert result[1][0] is tensor1
        assert result[1][1] is tensor2

    def test_get_all_local_class_features_different_devices(self) -> None:
        """Test get_all_local_class_features with tensors on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        features1 = Features()
        features1.local_features = {1: [torch.randn(2, 4).cuda()]}

        features2 = Features()
        features2.local_features = {1: [torch.randn(1, 4)]}  # CPU

        result = FeatureSelector.get_all_local_class_features([features1, features2])

        assert isinstance(result, dict)
        assert 1 in result
        assert len(result[1]) == 2

    def test_inheritance_works_correctly(self) -> None:
        """Test that concrete classes inherit from FeatureSelector correctly."""
        from getiprompt.processes.feature_selectors.all_features import AllFeaturesSelector
        from getiprompt.processes.feature_selectors.average_features import AverageFeatures
        from getiprompt.processes.feature_selectors.cluster_features import ClusterFeatures

        assert issubclass(AllFeaturesSelector, FeatureSelector)
        assert issubclass(AverageFeatures, FeatureSelector)
        assert issubclass(ClusterFeatures, FeatureSelector)

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

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_get_all_local_class_features_with_empty_feature_lists(self) -> None:
        """Test get_all_local_class_features with empty feature lists."""
        features = Features()
        features.local_features = {1: [], 2: []}

        result = FeatureSelector.get_all_local_class_features([features])

        assert isinstance(result, dict)
        assert len(result) == 2
        assert 1 in result
        assert 2 in result
        assert len(result[1]) == 0
        assert len(result[2]) == 0
