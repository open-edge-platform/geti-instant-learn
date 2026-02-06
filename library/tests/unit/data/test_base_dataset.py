# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for InstantLearnDataset base class."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
import torch

from instantlearn.data.base import Batch, Dataset, Sample


class MockDataset(Dataset):
    """Mock dataset for testing InstantLearnDataset base class."""

    def _load_dataframe(self) -> pl.DataFrame:
        """Load mock DataFrame."""
        return pl.DataFrame({
            "image_id": ["img1", "img2", "img3"],
            "image_path": ["/path/to/img1.jpg", "/path/to/img2.jpg", "/path/to/img3.jpg"],
            "categories": [["cat"], ["dog"], ["car"]],
            "category_ids": [[0], [1], [2]],
            "is_reference": [[True], [False], [True]],
            "n_shot": [[0], [-1], [0]],
            "mask_paths": [["/path/to/mask1.png"], ["/path/to/mask2.png"], ["/path/to/mask3.png"]],
        })

    def _load_masks(self, raw_sample: dict[str, Any]) -> torch.Tensor | None:
        """Load mock masks."""
        mask_paths = raw_sample.get("mask_paths", [])
        if not mask_paths:
            return None

        # Create dummy masks based on number of mask paths
        num_masks = len(mask_paths)
        return torch.randint(0, 2, (num_masks, 100, 100), dtype=torch.uint8)

    def load_test_dataframe(self) -> pl.DataFrame:
        """Helper method to load test dataframe."""
        return self._load_dataframe()


class TestInstantLearnDatasetBasic:
    """Test InstantLearnDataset basic functionality."""

    def test_dataset_initialization(self) -> None:
        """Test dataset initialization."""
        dataset = MockDataset(n_shots=2)
        assert dataset.n_shots == 2
        # Test that df is not initialized
        with pytest.raises(RuntimeError, match="Dataset not initialized"):
            _ = dataset.df

    def test_dataset_name_property(self) -> None:
        """Test dataset name property."""
        dataset = MockDataset()
        assert dataset.name == "Mock"

    def test_dataset_df_property_not_initialized(self) -> None:
        """Test that accessing df before initialization raises error."""
        dataset = MockDataset()
        with pytest.raises(RuntimeError, match="Dataset not initialized"):
            _ = dataset.df

    def test_dataset_df_setter_valid_dataframe(self) -> None:
        """Test setting valid DataFrame."""
        dataset = MockDataset()
        test_dataframe = pl.DataFrame({
            "image_path": ["/path/to/img1.jpg"],
            "categories": [["cat"]],
            "category_ids": [[0]],
            "is_reference": [[True]],
            "n_shot": [[0]],
        })

        dataset.df = test_dataframe
        assert len(dataset) == 1

    def test_dataset_df_setter_missing_columns(self) -> None:
        """Test that setting DataFrame with missing columns raises error."""
        dataset = MockDataset()
        invalid_dataframe = pl.DataFrame({
            "image_path": ["/path/to/img1.jpg"],
            "categories": [["cat"]],
            # Missing required columns
        })

        with pytest.raises(ValueError, match="DataFrame missing required columns"):
            dataset.df = invalid_dataframe

    def test_dataset_length(self) -> None:
        """Test dataset length."""
        dataset = MockDataset()
        dataset.df = dataset.load_test_dataframe()
        assert len(dataset) == 3


class TestInstantLearnDatasetCore:
    """Test InstantLearnDataset core functionality."""

    @patch("instantlearn.data.base.base.read_image")
    def test_dataset_getitem(self, mock_read_image: MagicMock) -> None:
        """Test dataset __getitem__ method."""
        # Mock image reading
        mock_read_image.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        dataset = MockDataset()
        dataset.df = dataset.load_test_dataframe()

        sample = dataset[0]

        assert isinstance(sample, Sample)
        assert sample.image.shape == (224, 224, 3)
        assert sample.image_path == "/path/to/img1.jpg"
        assert sample.categories == ["cat"]
        assert sample.category_ids.tolist() == [0]
        assert sample.is_reference == [True]
        assert sample.n_shot == [0]
        assert sample.mask_paths == ["/path/to/mask1.png"]
        assert sample.masks is not None
        assert sample.masks.shape[0] == 1  # One mask

    @patch("instantlearn.data.base.base.read_image")
    def test_dataset_getitem_with_bboxes(self, mock_read_image: MagicMock) -> None:
        """Test dataset __getitem__ with bboxes."""
        # Mock image reading
        mock_read_image.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        # Create dataset with bboxes
        bbox_dataframe = pl.DataFrame({
            "image_path": ["/path/to/img1.jpg"],
            "categories": [["cat"]],
            "category_ids": [[0]],
            "is_reference": [[True]],
            "n_shot": [[0]],
            "bboxes": [[[10, 20, 100, 120]]],
        })

        dataset = MockDataset()
        dataset.df = bbox_dataframe

        sample = dataset[0]

        assert sample.bboxes is not None
        assert sample.bboxes.shape == (1, 4)
        assert sample.bboxes[0, 0] == 10
        assert sample.bboxes[0, 1] == 20
        assert sample.bboxes[0, 2] == 100
        assert sample.bboxes[0, 3] == 120


class TestInstantLearnDatasetProperties:
    """Test InstantLearnDataset properties and metadata."""

    def test_dataset_categories_property(self) -> None:
        """Test dataset categories property."""
        dataset = MockDataset()
        dataset.df = dataset.load_test_dataframe()

        categories = dataset.categories
        assert sorted(categories) == ["car", "cat", "dog"]

    def test_dataset_category_ids_property(self) -> None:
        """Test dataset category_ids property."""
        dataset = MockDataset()
        dataset.df = dataset.load_test_dataframe()

        category_ids = dataset.category_ids
        assert sorted(category_ids) == [0, 1, 2]

    def test_dataset_num_categories_property(self) -> None:
        """Test dataset num_categories property."""
        dataset = MockDataset()
        dataset.df = dataset.load_test_dataframe()

        num_categories = dataset.num_categories
        assert num_categories == 3

    def test_dataset_get_category_name(self) -> None:
        """Test get_category_name method."""
        dataset = MockDataset()
        dataset.df = dataset.load_test_dataframe()

        assert dataset.get_category_name(0) == "cat"
        assert dataset.get_category_name(1) == "dog"
        assert dataset.get_category_name(2) == "car"

    def test_dataset_get_category_name_not_found(self) -> None:
        """Test get_category_name with non-existent ID."""
        dataset = MockDataset()
        dataset.df = dataset.load_test_dataframe()

        with pytest.raises(ValueError, match="Category ID 999 not found"):
            dataset.get_category_name(999)

    def test_dataset_get_category_id(self) -> None:
        """Test get_category_id method."""
        dataset = MockDataset()
        dataset.df = dataset.load_test_dataframe()

        assert dataset.get_category_id("cat") == 0
        assert dataset.get_category_id("dog") == 1
        assert dataset.get_category_id("car") == 2

    def test_dataset_get_category_id_not_found(self) -> None:
        """Test get_category_id with non-existent name."""
        dataset = MockDataset()
        dataset.df = dataset.load_test_dataframe()

        with pytest.raises(ValueError, match="Category 'unknown' not found"):
            dataset.get_category_id("unknown")


class TestInstantLearnDatasetFiltering:
    """Test InstantLearnDataset filtering functionality."""

    def test_dataset_get_reference_samples_df(self) -> None:
        """Test get_reference_samples_df method."""
        dataset = MockDataset()
        dataset.df = dataset.load_test_dataframe()

        ref_df = dataset.get_reference_samples_df()
        assert len(ref_df) == 2  # Two images have reference instances

        # Test filtering by category
        cat_ref_df = dataset.get_reference_samples_df(category="cat")
        assert len(cat_ref_df) == 1

        car_ref_df = dataset.get_reference_samples_df(category="car")
        assert len(car_ref_df) == 1

        # Test non-existent category
        unknown_ref_df = dataset.get_reference_samples_df(category="unknown")
        assert len(unknown_ref_df) == 0

    def test_dataset_get_target_samples_df(self) -> None:
        """Test get_target_samples_df method."""
        dataset = MockDataset()
        dataset.df = dataset.load_test_dataframe()

        target_df = dataset.get_target_samples_df()
        assert len(target_df) == 1  # One image has only target instances

        # Test filtering by category
        dog_target_df = dataset.get_target_samples_df(category="dog")
        assert len(dog_target_df) == 1

        # Test non-existent category
        unknown_target_df = dataset.get_target_samples_df(category="unknown")
        assert len(unknown_target_df) == 0

    def test_dataset_get_reference_dataset(self) -> None:
        """Test get_reference_dataset method."""
        dataset = MockDataset()
        dataset.df = dataset.load_test_dataframe()

        ref_dataset = dataset.get_reference_dataset()
        assert len(ref_dataset) == 2
        assert ref_dataset.name == "Mock"

        # Test filtering by category
        cat_ref_dataset = dataset.get_reference_dataset(category="cat")
        assert len(cat_ref_dataset) == 1

    def test_dataset_get_target_dataset(self) -> None:
        """Test get_target_dataset method."""
        dataset = MockDataset()
        dataset.df = dataset.load_test_dataframe()

        target_dataset = dataset.get_target_dataset()
        assert len(target_dataset) == 1
        assert target_dataset.name == "Mock"

        # Test filtering by category
        dog_target_dataset = dataset.get_target_dataset(category="dog")
        assert len(dog_target_dataset) == 1


class TestInstantLearnDatasetOperations:
    """Test InstantLearnDataset operations."""

    def test_dataset_subsample(self) -> None:
        """Test subsample method."""
        dataset = MockDataset()
        dataset.df = dataset.load_test_dataframe()

        # Test subsample
        subset = dataset.subsample([0, 2])
        assert len(subset) == 2
        assert subset.name == "Mock"

        # Test inplace subsample
        original_len = len(dataset)
        dataset.subsample([0, 1], inplace=True)
        assert len(dataset) == 2
        assert original_len == 3  # Original was 3

    def test_dataset_subsample_duplicate_indices(self) -> None:
        """Test subsample with duplicate indices raises error."""
        dataset = MockDataset()
        dataset.df = dataset.load_test_dataframe()

        with pytest.raises(ValueError, match="No duplicates allowed in indices"):
            dataset.subsample([0, 0])

    def test_dataset_add(self) -> None:
        """Test dataset concatenation."""
        dataset1 = MockDataset()
        dataset1.df = dataset1.load_test_dataframe()

        dataset2 = MockDataset()
        dataset2.df = dataset2.load_test_dataframe()

        combined = dataset1 + dataset2
        assert len(combined) == 6  # 3 + 3
        assert combined.name == "Mock"

    def test_dataset_add_different_types(self) -> None:
        """Test concatenating datasets of different types raises error."""
        dataset1 = MockDataset()
        dataset1.df = dataset1.load_test_dataframe()

        class DifferentDataset(Dataset):
            def _load_dataframe(self) -> pl.DataFrame:
                return pl.DataFrame()

            def _load_masks(self, raw_sample: dict[str, Any]) -> torch.Tensor | None:
                _ = raw_sample  # Unused but required by interface
                return None

        dataset2 = DifferentDataset()
        dataset2.df = pl.DataFrame({
            "image_path": ["/path/to/img1.jpg"],
            "categories": [["cat"]],
            "category_ids": [[0]],
            "is_reference": [[True]],
            "n_shot": [[0]],
        })

        with pytest.raises(TypeError, match="Cannot concatenate datasets that are not of the same type"):
            _ = dataset1 + dataset2

    def test_dataset_collate_fn_property(self) -> None:
        """Test collate_fn property."""
        dataset = MockDataset()
        assert dataset.collate_fn == Batch.collate


class TestInstantLearnDatasetAdvanced:
    """Test InstantLearnDataset advanced functionality."""

    @patch("instantlearn.data.base.base.read_image")
    def test_dataset_multi_instance_support(self, mock_read_image: MagicMock) -> None:
        """Test dataset with multi-instance data."""
        # Mock image reading
        mock_read_image.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        # Create multi-instance dataset
        multi_instance_dataframe = pl.DataFrame({
            "image_path": ["/path/to/img1.jpg"],
            "categories": [["person", "car", "dog"]],
            "category_ids": [[0, 1, 2]],
            "is_reference": [[True, False, True]],
            "n_shot": [[0, -1, 1]],
            "mask_paths": [["/path/to/mask1.png", "/path/to/mask2.png", "/path/to/mask3.png"]],
        })

        dataset = MockDataset()
        dataset.df = multi_instance_dataframe

        sample = dataset[0]

        assert sample.categories == ["person", "car", "dog"]
        assert sample.category_ids.tolist() == [0, 1, 2]
        assert sample.is_reference == [True, False, True]
        assert sample.n_shot == [0, -1, 1]
        assert sample.mask_paths == ["/path/to/mask1.png", "/path/to/mask2.png", "/path/to/mask3.png"]
        assert sample.masks is not None
        assert sample.masks.shape[0] == 3  # Three masks

    def test_dataset_abstract_methods(self) -> None:
        """Test that abstract methods must be implemented."""

        class IncompleteDataset(Dataset):
            pass

        # Should raise error when trying to instantiate
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteDataset()

    def test_dataset_empty_dataframe(self) -> None:
        """Test dataset with empty DataFrame."""
        dataset = MockDataset()
        dataset.df = pl.DataFrame({
            "image_path": [],
            "categories": [],
            "category_ids": [],
            "is_reference": [],
            "n_shot": [],
        })

        assert len(dataset) == 0

        # Test properties with empty dataset
        assert dataset.categories == []
        assert dataset.category_ids == []
        assert dataset.num_categories == 0

    @patch("instantlearn.data.base.base.read_image")
    def test_dataset_missing_optional_columns(self, mock_read_image: MagicMock) -> None:
        """Test dataset with missing optional columns."""
        # Mock image reading
        mock_read_image.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        dataset = MockDataset()
        minimal_dataframe = pl.DataFrame({
            "image_path": ["/path/to/img1.jpg"],
            "categories": [["cat"]],
            "category_ids": [[0]],
            "is_reference": [[True]],
            "n_shot": [[0]],
            # No mask_paths, bboxes, etc.
        })

        dataset.df = minimal_dataframe
        assert len(dataset) == 1

        # Test that missing columns are handled gracefully
        sample = dataset[0]
        assert sample.mask_paths is None
        assert sample.bboxes is None
