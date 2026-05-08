# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for COCO dataset functionality."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
import torch
from pycocotools import mask as mask_utils

from instantlearn.data.base import Batch, Dataset, Sample
from instantlearn.data.coco import COCODataset
from instantlearn.data.lvis import LVISAnnotationMode


class TestCOCODatasetMock:
    """Test COCODataset via a mock that bypasses filesystem / COCO API."""

    @pytest.fixture
    def mock_coco_dataframe(self) -> pl.DataFrame:
        """Create a mock DataFrame mimicking COCODataset._load_dataframe output (SEMANTIC mode)."""
        return pl.DataFrame({
            "image_id": [1, 1, 2, 3],
            "image_path": ["/dummy/img_001.jpg", "/dummy/img_001.jpg", "/dummy/img_002.jpg", "/dummy/img_003.jpg"],
            "categories": [["cat"], ["dog"], ["cat"], ["dog"]],
            "category_ids": [[1], [2], [1], [2]],
            "segmentations": [
                [[[10, 10, 50, 10, 50, 50, 10, 50]]],
                [[[60, 60, 90, 60, 90, 90, 60, 90]]],
                [[[20, 20, 80, 20, 80, 80, 20, 80]]],
                [[[5, 5, 40, 5, 40, 40, 5, 40]]],
            ],
            "bboxes": [None, None, None, None],
            "is_reference": [[True], [True], [False], [False]],
            "n_shot": [[0], [0], [-1], [-1]],
            "img_dim": [(100, 100), (100, 100), (100, 100), (100, 100)],
        })

    @pytest.fixture
    def mock_coco_dataset(self, mock_coco_dataframe: pl.DataFrame) -> Dataset:
        """Create a mock COCO dataset that skips real file I/O."""

        class MockCOCODataset(Dataset):
            def _load_dataframe(self) -> pl.DataFrame:
                return mock_coco_dataframe

            def _load_masks(self, raw_sample: dict[str, Any]) -> np.ndarray:
                segmentations = raw_sample.get("segmentations", [])
                if not segmentations:
                    return np.zeros((0, 100, 100), dtype=np.uint8)
                num_masks = len(segmentations)
                return np.random.default_rng(42).integers(0, 2, (num_masks, 100, 100), dtype=np.uint8)

        dataset = MockCOCODataset()
        dataset.df = mock_coco_dataframe
        return dataset

    def test_dataset_length(self, mock_coco_dataset: Dataset) -> None:
        """Dataset length matches number of rows."""
        assert len(mock_coco_dataset) == 4

    def test_categories_property(self, mock_coco_dataset: Dataset) -> None:
        """Unique categories are returned."""
        cats = mock_coco_dataset.categories
        assert set(cats) == {"cat", "dog"}

    def test_num_categories(self, mock_coco_dataset: Dataset) -> None:
        """Number of unique categories is correct."""
        assert mock_coco_dataset.num_categories == 2

    def test_reference_filtering(self, mock_coco_dataset: Dataset) -> None:
        """Reference rows are correctly filtered."""
        ref_df = mock_coco_dataset.get_reference_samples_df()
        assert len(ref_df) == 2

    def test_target_filtering(self, mock_coco_dataset: Dataset) -> None:
        """Target rows are correctly filtered."""
        target_df = mock_coco_dataset.get_target_samples_df()
        assert len(target_df) == 2

    def test_category_reference_filtering(self, mock_coco_dataset: Dataset) -> None:
        """Reference rows can be filtered by category."""
        cat_ref = mock_coco_dataset.get_reference_samples_df(category="cat")
        assert len(cat_ref) == 1
        dog_ref = mock_coco_dataset.get_reference_samples_df(category="dog")
        assert len(dog_ref) == 1

    @patch("instantlearn.data.base.base.read_image")
    def test_sample_creation(self, mock_read_image: MagicMock, mock_coco_dataset: Dataset) -> None:
        """Samples are created with correct structure."""
        mock_read_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        sample = mock_coco_dataset[0]
        assert isinstance(sample, Sample)
        assert len(sample.categories) == 1
        assert sample.categories == ["cat"]
        assert sample.masks is not None
        assert sample.masks.shape[0] == 1

    @patch("instantlearn.data.base.base.read_image")
    def test_sample_metadata(self, mock_read_image: MagicMock, mock_coco_dataset: Dataset) -> None:
        """Sample metadata fields are correct."""
        mock_read_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        sample = mock_coco_dataset[0]
        assert sample.is_reference == [True]
        assert sample.n_shot == [0]

    @patch("instantlearn.data.base.base.read_image")
    def test_batch_creation(self, mock_read_image: MagicMock, mock_coco_dataset: Dataset) -> None:
        """Batch collation preserves multi-sample structure."""
        mock_read_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        samples = [mock_coco_dataset[i] for i in range(len(mock_coco_dataset))]
        batch = Batch.collate(samples)

        assert isinstance(batch, Batch)
        assert len(batch) == 4
        assert len(batch.categories) == 4
        assert len(batch.images) == 4

    @patch("instantlearn.data.base.base.read_image")
    def test_data_consistency(self, mock_read_image: MagicMock, mock_coco_dataset: Dataset) -> None:
        """All samples have consistent metadata lengths."""
        mock_read_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        for i in range(len(mock_coco_dataset)):
            sample = mock_coco_dataset[i]
            assert len(sample.categories) == len(sample.category_ids)
            assert len(sample.categories) == len(sample.is_reference)
            assert len(sample.categories) == len(sample.n_shot)


class TestCOCODatasetMaskDecoding:
    """Test COCODataset mask decoding methods directly."""

    def test_decode_single_polygon(self) -> None:
        """Polygon segmentation decodes to a binary mask."""
        polygon = [[10, 10, 90, 10, 90, 90, 10, 90]]
        mask = COCODataset._decode_single(polygon, h=100, w=100)  # noqa: SLF001
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool
        assert mask.shape == (100, 100)
        assert mask.any()

    def test_decode_single_rle(self) -> None:
        """RLE segmentation decodes to a binary mask."""
        # Create a valid RLE from a polygon
        rle = mask_utils.frPyObjects([[10, 10, 90, 10, 90, 90, 10, 90]], 100, 100)
        merged_rle = mask_utils.merge(rle)
        mask = COCODataset._decode_single(merged_rle, h=100, w=100)  # noqa: SLF001
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool
        assert mask.shape == (100, 100)
        assert mask.any()

    def test_decode_single_invalid_type(self) -> None:
        """Invalid segmentation type raises TypeError."""
        with pytest.raises(TypeError, match="Unknown segmentation format"):
            COCODataset._decode_single(12345, h=100, w=100)  # noqa: SLF001

    def test_load_masks_semantic_mode(self) -> None:
        """SEMANTIC mode merges multiple polygons into one mask."""
        dataset = COCODataset.__new__(COCODataset)
        dataset.annotation_mode = LVISAnnotationMode.SEMANTIC

        raw_sample = {
            "segmentations": [
                [[10, 10, 40, 10, 40, 40, 10, 40]],
                [[60, 60, 90, 60, 90, 90, 60, 90]],
            ],
            "img_dim": (100, 100),
        }
        masks = dataset._load_masks(raw_sample)  # noqa: SLF001
        assert masks is not None
        assert masks.shape == (1, 100, 100)
        assert masks.any()

    def test_load_masks_instance_mode(self) -> None:
        """INSTANCE mode keeps separate masks per instance."""
        dataset = COCODataset.__new__(COCODataset)
        dataset.annotation_mode = LVISAnnotationMode.INSTANCE

        raw_sample = {
            "segmentations": [
                [[10, 10, 40, 10, 40, 40, 10, 40]],
                [[60, 60, 90, 60, 90, 90, 60, 90]],
            ],
            "img_dim": (100, 100),
        }
        masks = dataset._load_masks(raw_sample)  # noqa: SLF001
        assert masks is not None
        assert masks.shape == (2, 100, 100)

    def test_load_masks_empty(self) -> None:
        """Empty segmentations return None."""
        dataset = COCODataset.__new__(COCODataset)
        dataset.annotation_mode = LVISAnnotationMode.SEMANTIC

        raw_sample = {"segmentations": [], "img_dim": (100, 100)}
        assert dataset._load_masks(raw_sample) is None  # noqa: SLF001

        raw_sample_no_key = {"img_dim": (100, 100)}
        assert dataset._load_masks(raw_sample_no_key) is None  # noqa: SLF001
