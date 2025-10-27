# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PerSeg dataset functionality."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from getiprompt.data.base import Batch, Dataset, Sample


class TestPerSegDataset:
    """Test PerSeg dataset functionality with dummy data."""

    @pytest.fixture
    def mock_perseg_dataframe(self) -> pl.DataFrame:
        """Create a mock DataFrame with PerSeg-style single-instance data."""
        return pl.DataFrame({
            "image_id": ["perseg_001", "perseg_002", "perseg_003"],
            "image_path": ["/dummy/perseg_001.jpg", "/dummy/perseg_002.jpg", "/dummy/perseg_003.jpg"],
            "categories": [["person"], ["car"], ["dog"]],
            "category_ids": [[0], [1], [2]],
            "is_reference": [[True], [False], [True]],
            "n_shot": [[0], [-1], [0]],
            "mask_paths": [
                ["/dummy/mask_001.png"],
                ["/dummy/mask_002.png"],
                ["/dummy/mask_003.png"],
            ],
        })

    @pytest.fixture
    def mock_perseg_dataset(self, mock_perseg_dataframe: pl.DataFrame) -> Dataset:
        """Create a mock PerSeg dataset."""

        class MockPerSegDataset(Dataset):
            def _load_dataframe(self) -> pl.DataFrame:
                return mock_perseg_dataframe

            def _load_masks(self, raw_sample: dict[str, Any]) -> np.ndarray:
                # Simulate loading masks from mask paths
                mask_paths = raw_sample.get("mask_paths", [])
                if not mask_paths:
                    return np.zeros((0, 100, 100), dtype=np.uint8)

                # Create dummy masks based on number of mask paths
                num_masks = len(mask_paths)
                return np.random.default_rng(42).integers(0, 2, (num_masks, 100, 100), dtype=np.uint8)

        dataset = MockPerSegDataset()
        dataset.df = mock_perseg_dataframe
        return dataset

    @patch("getiprompt.data.base.base.read_image")
    def test_perseg_sample_creation(
        self,
        mock_read_image: MagicMock,
        mock_perseg_dataset: Dataset,
    ) -> None:
        """Test PerSeg sample creation with single-instance data."""
        # Mock image reading
        mock_read_image.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        # Test that the dataset recognizes single-instance structure
        assert len(mock_perseg_dataset) == 3

        # Test that samples are created correctly
        sample = mock_perseg_dataset[0]
        assert isinstance(sample, Sample)
        assert len(sample.categories) == 1  # Single-instance
        assert len(sample.category_ids) == 1
        assert len(sample.is_reference) == 1
        assert len(sample.n_shot) == 1
        assert sample.masks is not None
        assert sample.masks.shape[0] == 1  # One mask for first image

    @patch("getiprompt.data.base.base.read_image")
    def test_perseg_batch_creation(
        self,
        mock_read_image: MagicMock,
        mock_perseg_dataset: Dataset,
    ) -> None:
        """Test PerSeg batch creation with single-instance data."""
        # Mock image reading
        mock_read_image.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        # Create a batch
        samples = [mock_perseg_dataset[i] for i in range(len(mock_perseg_dataset))]
        batch = Batch.collate(samples)

        assert isinstance(batch, Batch)
        assert len(batch) == 3

        # Test batch properties with single-instance data
        assert len(batch.categories) == 3
        assert len(batch.category_ids) == 3
        assert len(batch.is_reference) == 3

        # Check that single-instance structure is preserved
        assert len(batch.categories[0]) == 1  # First image has 1 instance
        assert len(batch.categories[1]) == 1  # Second image has 1 instance
        assert len(batch.categories[2]) == 1  # Third image has 1 instance

    def test_perseg_reference_filtering(self, mock_perseg_dataset: Dataset) -> None:
        """Test PerSeg reference filtering with single-instance data."""
        # Test get_reference_samples_df
        ref_df = mock_perseg_dataset.get_reference_samples_df()
        assert len(ref_df) == 2  # Two images have reference instances

        # Test get_target_samples_df
        target_df = mock_perseg_dataset.get_target_samples_df()
        assert len(target_df) == 1  # One image has only target instances

    def test_perseg_category_filtering(self, mock_perseg_dataset: Dataset) -> None:
        """Test PerSeg category filtering with single-instance data."""
        # Test filtering by category
        person_df = mock_perseg_dataset.get_reference_samples_df(category="person")
        assert len(person_df) == 1  # Only one image has "person" reference

        # "car" is not a reference (is_reference: [False]), so it won't be in reference samples
        car_df = mock_perseg_dataset.get_reference_samples_df(category="car")
        assert len(car_df) == 0  # "car" is not a reference

        dog_df = mock_perseg_dataset.get_reference_samples_df(category="dog")
        assert len(dog_df) == 1  # Only one image has "dog" reference

    @patch("getiprompt.data.base.base.read_image")
    def test_perseg_sample_metadata(
        self,
        mock_read_image: MagicMock,
        mock_perseg_dataset: Dataset,
    ) -> None:
        """Test PerSeg sample metadata with single-instance data."""
        # Mock image reading
        mock_read_image.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        sample = mock_perseg_dataset[0]

        # Test metadata fields
        assert sample.categories == ["person"]
        assert sample.category_ids.tolist() == [0]
        assert sample.is_reference == [True]
        assert sample.n_shot == [0]

    @patch("getiprompt.data.base.base.read_image")
    def test_perseg_batch_properties(
        self,
        mock_read_image: MagicMock,
        mock_perseg_dataset: Dataset,
    ) -> None:
        """Test PerSeg batch properties with single-instance data."""
        # Mock image reading
        mock_read_image.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        # Create a batch
        samples = [mock_perseg_dataset[i] for i in range(len(mock_perseg_dataset))]
        batch = Batch.collate(samples)

        # Test batch properties
        assert batch.categories == [["person"], ["car"], ["dog"]]
        assert [ids.tolist() for ids in batch.category_ids] == [[0], [1], [2]]
        assert batch.is_reference == [[True], [False], [True]]
        assert batch.n_shot == [[0], [-1], [0]]

    @patch("getiprompt.data.base.base.read_image")
    def test_perseg_data_consistency(
        self,
        mock_read_image: MagicMock,
        mock_perseg_dataset: Dataset,
    ) -> None:
        """Test PerSeg data consistency for single-instance handling."""
        # Mock image reading
        mock_read_image.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        # Test that all samples have consistent structure
        for i in range(len(mock_perseg_dataset)):
            sample = mock_perseg_dataset[i]

            # All metadata lists should have the same length
            assert len(sample.categories) == len(sample.category_ids)
            assert len(sample.categories) == len(sample.is_reference)
            assert len(sample.categories) == len(sample.n_shot)

            # All values should be lists (except category_ids which can be numpy arrays)
            assert isinstance(sample.categories, list)
            assert hasattr(sample.category_ids, "tolist")  # numpy array or list
            assert isinstance(sample.is_reference, list)
            assert isinstance(sample.n_shot, list)

    @patch("getiprompt.data.base.base.read_image")
    def test_perseg_batch_tensor_conversion(
        self,
        mock_read_image: MagicMock,
        mock_perseg_dataset: Dataset,
    ) -> None:
        """Test PerSeg tensor conversion in batches with single-instance data."""
        # Mock image reading
        mock_read_image.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        # Create a batch
        samples = [mock_perseg_dataset[i] for i in range(len(mock_perseg_dataset))]
        batch = Batch.collate(samples)

        # Test tensor conversion - check that category_ids are properly converted
        assert len(batch.category_ids) == 3  # One per image

        # Each tensor should have exactly one instance (single-instance)
        assert len(batch.category_ids[0]) == 1  # First image has 1 instance
        assert len(batch.category_ids[1]) == 1  # Second image has 1 instance
        assert len(batch.category_ids[2]) == 1  # Third image has 1 instance

    @patch("getiprompt.data.base.base.read_image")
    def test_perseg_sample_loading(
        self,
        mock_read_image: MagicMock,
        mock_perseg_dataset: Dataset,
    ) -> None:
        """Test PerSeg sample loading with single-instance data."""
        # Mock image reading
        mock_read_image.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        # Test sample loading
        sample = mock_perseg_dataset[0]
        assert sample.image is not None
        assert sample.image.shape == (224, 224, 3)
        assert sample.masks is not None
        # The first image has 1 mask path, so should have 1 mask
        assert sample.masks.shape[0] == 1  # One mask for first image

    def test_perseg_empty_dataset(self) -> None:
        """Test PerSeg edge cases with empty dataset."""
        # Test with empty single-instance data
        empty_df = pl.DataFrame({
            "image_id": [],
            "image_path": [],
            "categories": [],
            "category_ids": [],
            "is_reference": [],
            "n_shot": [],
            "mask_paths": [],
        })

        class MockEmptyDataset(Dataset):
            def _load_dataframe(self) -> pl.DataFrame:
                return empty_df

            def _load_masks(self, raw_sample: dict[str, Any]) -> np.ndarray:
                _ = raw_sample  # Unused but required by interface
                return np.zeros((0, 100, 100), dtype=np.uint8)

        dataset = MockEmptyDataset()
        dataset.df = empty_df
        assert len(dataset) == 0
