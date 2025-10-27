# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LVIS dataset functionality."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from getiprompt.data.base import Batch, Dataset, Sample


class TestLVISDataset:
    """Test LVIS dataset functionality with dummy data."""

    @pytest.fixture
    def mock_lvis_dataframe(self) -> pl.DataFrame:
        """Create a mock DataFrame with LVIS-style multi-instance data."""
        return pl.DataFrame({
            "image_id": ["lvis_001", "lvis_002", "lvis_003"],
            "image_path": ["/dummy/lvis_001.jpg", "/dummy/lvis_002.jpg", "/dummy/lvis_003.jpg"],
            "categories": [["person", "car"], ["dog"], ["person", "bike", "car"]],
            "category_ids": [[0, 1], [2], [0, 3, 1]],
            "is_reference": [[True, False], [True], [False, False, False]],
            "n_shot": [[0, -1], [0], [-1, -1, -1]],
            "segmentations": [
                [{"size": [100, 100], "counts": "dummy_rle_1"}, {"size": [100, 100], "counts": "dummy_rle_2"}],
                [{"size": [100, 100], "counts": "dummy_rle_3"}],
                [
                    {"size": [100, 100], "counts": "dummy_rle_4"},
                    {"size": [100, 100], "counts": "dummy_rle_5"},
                    {"size": [100, 100], "counts": "dummy_rle_6"},
                ],
            ],
        })

    @pytest.fixture
    def mock_lvis_dataset(self, mock_lvis_dataframe: pl.DataFrame) -> Dataset:
        """Create a mock LVIS dataset."""

        class MockLVISDataset(Dataset):
            def _load_dataframe(self) -> pl.DataFrame:
                return mock_lvis_dataframe

            def _load_masks(self, raw_sample: dict[str, Any]) -> np.ndarray:
                # Simulate loading masks from RLE segmentations
                segmentations = raw_sample.get("segmentations", [])
                if not segmentations:
                    return np.zeros((0, 100, 100), dtype=np.uint8)

                # Create dummy masks based on number of segmentations
                num_masks = len(segmentations)
                return np.random.default_rng(42).integers(0, 2, (num_masks, 100, 100), dtype=np.uint8)

        dataset = MockLVISDataset()
        dataset.df = mock_lvis_dataframe
        return dataset

    @patch("getiprompt.data.base.base.read_image")
    def test_lvis_sample_creation(
        self,
        mock_read_image: MagicMock,
        mock_lvis_dataset: Dataset,
    ) -> None:
        """Test LVIS sample creation with multi-instance data."""
        # Mock image reading
        mock_read_image.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        # Test that the dataset recognizes multi-instance structure
        assert len(mock_lvis_dataset) == 3

        # Test that samples are created correctly
        sample = mock_lvis_dataset[0]
        assert isinstance(sample, Sample)
        assert len(sample.categories) == 2  # Multi-instance
        assert len(sample.category_ids) == 2
        assert len(sample.is_reference) == 2
        assert len(sample.n_shot) == 2
        assert sample.masks is not None
        assert sample.masks.shape[0] == 2  # Two masks for first image

    @patch("getiprompt.data.base.base.read_image")
    def test_lvis_batch_creation(
        self,
        mock_read_image: MagicMock,
        mock_lvis_dataset: Dataset,
    ) -> None:
        """Test LVIS batch creation with multi-instance data."""
        # Mock image reading
        mock_read_image.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        # Create a batch
        samples = [mock_lvis_dataset[i] for i in range(len(mock_lvis_dataset))]
        batch = Batch.collate(samples)

        assert isinstance(batch, Batch)
        assert len(batch) == 3

        # Test batch properties with multi-instance data
        assert len(batch.categories) == 3
        assert len(batch.category_ids) == 3
        assert len(batch.is_reference) == 3

        # Check that multi-instance structure is preserved
        assert len(batch.categories[0]) == 2  # First image has 2 instances
        assert len(batch.categories[1]) == 1  # Second image has 1 instance
        assert len(batch.categories[2]) == 3  # Third image has 3 instances

    def test_lvis_reference_filtering(self, mock_lvis_dataset: Dataset) -> None:
        """Test LVIS reference filtering with multi-instance data."""
        # Test get_reference_samples_df
        ref_df = mock_lvis_dataset.get_reference_samples_df()
        assert len(ref_df) == 2  # Two images have reference instances

        # Test get_target_samples_df
        target_df = mock_lvis_dataset.get_target_samples_df()
        assert len(target_df) == 1  # One image has only target instances

    def test_lvis_category_filtering(self, mock_lvis_dataset: Dataset) -> None:
        """Test LVIS category filtering with multi-instance data."""
        # Test filtering by category
        person_df = mock_lvis_dataset.get_reference_samples_df(category="person")
        assert len(person_df) == 1  # Only one image has "person" reference

        car_df = mock_lvis_dataset.get_reference_samples_df(category="car")
        assert len(car_df) == 1  # Only one image has "car" reference

    @patch("getiprompt.data.base.base.read_image")
    def test_lvis_sample_metadata(
        self,
        mock_read_image: MagicMock,
        mock_lvis_dataset: Dataset,
    ) -> None:
        """Test LVIS sample metadata with multi-instance data."""
        # Mock image reading
        mock_read_image.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        sample = mock_lvis_dataset[0]

        # Test metadata fields
        assert sample.categories == ["person", "car"]
        assert sample.category_ids.tolist() == [0, 1]
        assert sample.is_reference == [True, False]
        assert sample.n_shot == [0, -1]

    @patch("getiprompt.data.base.base.read_image")
    def test_lvis_batch_properties(
        self,
        mock_read_image: MagicMock,
        mock_lvis_dataset: Dataset,
    ) -> None:
        """Test LVIS batch properties with multi-instance data."""
        # Mock image reading
        mock_read_image.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        # Create a batch
        samples = [mock_lvis_dataset[i] for i in range(len(mock_lvis_dataset))]
        batch = Batch.collate(samples)

        # Test batch properties
        assert batch.categories == [["person", "car"], ["dog"], ["person", "bike", "car"]]
        assert [ids.tolist() for ids in batch.category_ids] == [[0, 1], [2], [0, 3, 1]]
        assert batch.is_reference == [[True, False], [True], [False, False, False]]
        assert batch.n_shot == [[0, -1], [0], [-1, -1, -1]]

    @patch("getiprompt.data.base.base.read_image")
    def test_lvis_data_consistency(
        self,
        mock_read_image: MagicMock,
        mock_lvis_dataset: Dataset,
    ) -> None:
        """Test LVIS data consistency for multi-instance handling."""
        # Mock image reading
        mock_read_image.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        # Test that all samples have consistent structure
        for i in range(len(mock_lvis_dataset)):
            sample = mock_lvis_dataset[i]

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
    def test_lvis_batch_tensor_conversion(
        self,
        mock_read_image: MagicMock,
        mock_lvis_dataset: Dataset,
    ) -> None:
        """Test LVIS tensor conversion in batches with multi-instance data."""
        # Mock image reading
        mock_read_image.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        # Create a batch
        samples = [mock_lvis_dataset[i] for i in range(len(mock_lvis_dataset))]
        batch = Batch.collate(samples)

        # Test tensor conversion - check that category_ids are properly converted
        assert len(batch.category_ids) == 3  # One per image

        # Each tensor should have the correct number of instances
        assert len(batch.category_ids[0]) == 2  # First image has 2 instances
        assert len(batch.category_ids[1]) == 1  # Second image has 1 instance
        assert len(batch.category_ids[2]) == 3  # Third image has 3 instances

    @patch("getiprompt.data.base.base.read_image")
    def test_lvis_sample_loading(
        self,
        mock_read_image: MagicMock,
        mock_lvis_dataset: Dataset,
    ) -> None:
        """Test LVIS sample loading with multi-instance data."""
        # Mock image reading
        mock_read_image.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        # Test sample loading
        sample = mock_lvis_dataset[0]
        assert sample.image is not None
        assert sample.image.shape == (224, 224, 3)
        assert sample.masks is not None
        # The first image has 2 segmentations, so should have 2 masks
        assert sample.masks.shape[0] == 2  # Two masks for first image
