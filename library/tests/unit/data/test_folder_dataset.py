# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Folder dataset functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import torch

from instantlearn.data.base import Batch, Sample
from instantlearn.data.folder import FolderDataset


class TestFolderDatasetBasic:
    """Test FolderDataset basic functionality."""

    def test_dataset_initialization(self, tmp_path: Path) -> None:
        """Test FolderDataset initialization."""
        # Create a minimal folder structure
        images_dir = tmp_path / "images" / "apple"
        masks_dir = tmp_path / "masks" / "apple"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        # Create a test image and mask
        (images_dir / "1.jpg").touch()
        (masks_dir / "1.png").touch()

        dataset = FolderDataset(root=tmp_path, n_shots=1)
        assert dataset.n_shots == 1
        assert dataset.root == tmp_path
        assert dataset.images_dir == "images"
        assert dataset.masks_dir == "masks"

    def test_dataset_name_property(self, tmp_path: Path) -> None:
        """Test dataset name property."""
        # Create a minimal folder structure
        images_dir = tmp_path / "images" / "apple"
        masks_dir = tmp_path / "masks" / "apple"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)
        (images_dir / "1.jpg").touch()
        (masks_dir / "1.png").touch()

        dataset = FolderDataset(root=tmp_path)
        assert dataset.name == "Folder"

    def test_custom_images_masks_dirs(self, tmp_path: Path) -> None:
        """Test custom images_dir and masks_dir names."""
        # Create folder structure with custom names
        images_dir = tmp_path / "custom_images" / "apple"
        masks_dir = tmp_path / "custom_masks" / "apple"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)
        (images_dir / "1.jpg").touch()
        (masks_dir / "1.png").touch()

        dataset = FolderDataset(
            root=tmp_path,
            images_dir="custom_images",
            masks_dir="custom_masks",
        )
        assert dataset.images_dir == "custom_images"
        assert dataset.masks_dir == "custom_masks"
        assert len(dataset) > 0

    def test_dataset_length(self, tmp_path: Path) -> None:
        """Test dataset length."""
        # Create a folder structure with multiple images
        images_dir = tmp_path / "images" / "apple"
        masks_dir = tmp_path / "masks" / "apple"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        # Create 3 image-mask pairs
        for i in range(1, 4):
            (images_dir / f"{i}.jpg").touch()
            (masks_dir / f"{i}.png").touch()

        dataset = FolderDataset(root=tmp_path)
        assert len(dataset) == 3

    def test_missing_images_directory(self, tmp_path: Path) -> None:
        """Test error when images directory is missing."""
        with pytest.raises(FileNotFoundError, match="Images directory not found"):
            FolderDataset(root=tmp_path)

    def test_missing_masks_directory(self, tmp_path: Path) -> None:
        """Test error when masks directory is missing."""
        # Create images directory but not masks
        images_dir = tmp_path / "images" / "apple"
        images_dir.mkdir(parents=True)
        (images_dir / "1.jpg").touch()

        with pytest.raises(FileNotFoundError, match="Masks directory not found"):
            FolderDataset(root=tmp_path)

    def test_no_valid_categories(self, tmp_path: Path) -> None:
        """Test error when no valid categories are found."""
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)
        # Create directories but no category folders
        (images_dir / "file.txt").touch()

        with pytest.raises(ValueError, match="No valid categories found"):
            FolderDataset(root=tmp_path)

    def test_no_matching_pairs(self, tmp_path: Path) -> None:
        """Test error when no image-mask pairs are found."""
        images_dir = tmp_path / "images" / "apple"
        masks_dir = tmp_path / "masks" / "apple"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)
        # Create image but no matching mask
        (images_dir / "1.jpg").touch()

        with pytest.raises(ValueError, match="No valid image-mask pairs found"):
            FolderDataset(root=tmp_path)


class TestFolderDatasetWithFSS1000:
    """Test FolderDataset with actual fss-1000 dataset structure."""

    @pytest.fixture
    def fss1000_root(self) -> Path:
        """Return path to fss-1000 test dataset."""
        return Path(__file__).parent.parent.parent / "assets" / "fss-1000"

    def test_load_fss1000_dataset(self, fss1000_root: Path) -> None:
        """Test loading the fss-1000 dataset."""
        if not fss1000_root.exists():
            pytest.skip("fss-1000 dataset not found")

        dataset = FolderDataset(root=fss1000_root, n_shots=1)
        assert len(dataset) > 0
        assert len(dataset.categories) == 2

    def test_load_specific_categories(self, fss1000_root: Path) -> None:
        """Test loading specific categories."""
        if not fss1000_root.exists():
            pytest.skip("fss-1000 dataset not found")

        dataset = FolderDataset(
            root=fss1000_root,
            categories=["apple", "basketball"],
            n_shots=1,
        )
        assert len(dataset.categories) == 2
        assert "apple" in dataset.categories
        assert "basketball" in dataset.categories

    def test_n_shots_parameter(self, fss1000_root: Path) -> None:
        """Test n_shots parameter."""
        if not fss1000_root.exists():
            pytest.skip("fss-1000 dataset not found")

        dataset = FolderDataset(root=fss1000_root, categories=["apple", "basketball"], n_shots=3)

        # Check that first 3 samples are references
        apple_samples = dataset.df.filter(pl.col("categories").list.contains("apple"))
        assert apple_samples["is_reference"].list.first().sum() == 3  # First 3 are references

        # Check that first 3 samples are references
        basketball_samples = dataset.df.filter(pl.col("categories").list.contains("basketball"))
        assert basketball_samples["is_reference"].list.first().sum() == 3  # First 3 are references

    @patch("instantlearn.data.base.base.read_image")
    def test_sample_loading(self, mock_read_image: MagicMock, fss1000_root: Path) -> None:
        """Test sample loading from fss-1000."""
        if not fss1000_root.exists():
            pytest.skip("fss-1000 dataset not found")

        # Mock image reading
        mock_read_image.return_value = torch.zeros((3, 224, 224))

        dataset = FolderDataset(root=fss1000_root, categories=["apple"], n_shots=1)

        sample = dataset[0]
        assert isinstance(sample, Sample)
        assert sample.image is not None
        assert sample.categories is not None
        assert len(sample.categories) == 1  # Single-instance dataset
        assert sample.masks is not None
        assert sample.masks.shape[0] == 1  # One mask

    def test_category_properties(self, fss1000_root: Path) -> None:
        """Test category-related properties."""
        if not fss1000_root.exists():
            pytest.skip("fss-1000 dataset not found")

        dataset = FolderDataset(root=fss1000_root, categories=["apple", "basketball"])

        assert len(dataset.categories) == 2
        assert len(dataset.category_ids) == 2
        assert dataset.num_categories == 2

        # Test category name to ID mapping
        apple_id = dataset.get_category_id("apple")
        assert isinstance(apple_id, int)
        assert dataset.get_category_name(apple_id) == "apple"


class TestFolderDatasetSampleLoading:
    """Test FolderDataset sample loading functionality."""

    @pytest.fixture
    def mock_folder_dataset(self, tmp_path: Path) -> FolderDataset:
        """Create a mock folder dataset with test data."""
        # Create folder structure
        images_dir = tmp_path / "images" / "test_category"
        masks_dir = tmp_path / "masks" / "test_category"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        # Create 5 image-mask pairs
        for i in range(1, 6):
            (images_dir / f"{i}.jpg").touch()
            (masks_dir / f"{i}.png").touch()

        return FolderDataset(root=tmp_path, n_shots=2)

    @patch("instantlearn.data.base.base.read_image")
    @patch("instantlearn.data.folder.dataset.read_mask")
    def test_sample_creation(
        self,
        mock_read_mask: MagicMock,
        mock_read_image: MagicMock,
        mock_folder_dataset: FolderDataset,
    ) -> None:
        """Test sample creation."""
        # Mock image and mask reading
        mock_read_image.return_value = torch.zeros((3, 100, 100))
        mock_read_mask.return_value = torch.zeros((100, 100), dtype=torch.uint8)

        sample = mock_folder_dataset[0]

        assert isinstance(sample, Sample)
        assert sample.image is not None
        assert sample.masks is not None
        assert sample.masks.shape == (1, 100, 100)  # Single instance mask
        assert len(sample.categories) == 1
        assert len(sample.category_ids) == 1
        assert len(sample.is_reference) == 1
        assert len(sample.n_shot) == 1

    @patch("instantlearn.data.base.base.read_image")
    @patch("instantlearn.data.folder.dataset.read_mask")
    def test_sample_metadata(
        self,
        mock_read_mask: MagicMock,
        mock_read_image: MagicMock,
        mock_folder_dataset: FolderDataset,
    ) -> None:
        """Test sample metadata."""
        # Mock image and mask reading
        mock_read_image.return_value = torch.zeros((3, 100, 100))
        mock_read_mask.return_value = torch.zeros((100, 100), dtype=torch.uint8)

        # First sample should be a reference (n_shots=2, so first 2 are references)
        sample0 = mock_folder_dataset[0]
        assert sample0.is_reference == [True]
        assert sample0.n_shot == [0]

        # Second sample should also be a reference
        sample1 = mock_folder_dataset[1]
        assert sample1.is_reference == [True]
        assert sample1.n_shot == [1]

        # Third sample should be a target
        sample2 = mock_folder_dataset[2]
        assert sample2.is_reference == [False]
        assert sample2.n_shot == [-1]

    @patch("instantlearn.data.base.base.read_image")
    @patch("instantlearn.data.folder.dataset.read_mask")
    def test_sample_image_path(
        self,
        mock_read_mask: MagicMock,
        mock_read_image: MagicMock,
        mock_folder_dataset: FolderDataset,
    ) -> None:
        """Test that sample has correct image path."""
        # Mock image and mask reading
        mock_read_image.return_value = torch.zeros((3, 100, 100))
        mock_read_mask.return_value = torch.zeros((100, 100), dtype=torch.uint8)

        sample = mock_folder_dataset[0]
        assert sample.image_path is not None
        assert sample.image_path.endswith(".jpg")
        assert "test_category" in sample.image_path

    @patch("instantlearn.data.base.base.read_image")
    @patch("instantlearn.data.folder.dataset.read_mask")
    def test_sample_mask_paths(
        self,
        mock_read_mask: MagicMock,
        mock_read_image: MagicMock,
        mock_folder_dataset: FolderDataset,
    ) -> None:
        """Test that sample has correct mask paths."""
        # Mock image and mask reading
        mock_read_image.return_value = torch.zeros((3, 100, 100))
        mock_read_mask.return_value = torch.zeros((100, 100), dtype=torch.uint8)

        sample = mock_folder_dataset[0]
        assert sample.mask_paths is not None
        assert len(sample.mask_paths) == 1
        assert sample.mask_paths[0].endswith(".png")
        assert "test_category" in sample.mask_paths[0]


class TestFolderDatasetFiltering:
    """Test FolderDataset filtering functionality."""

    @pytest.fixture
    def multi_category_dataset(self, tmp_path: Path) -> FolderDataset:
        """Create a dataset with multiple categories."""
        categories = ["apple", "basketball", "orange"]

        for category in categories:
            images_dir = tmp_path / "images" / category
            masks_dir = tmp_path / "masks" / category
            images_dir.mkdir(parents=True)
            masks_dir.mkdir(parents=True)

            # Create 3 image-mask pairs per category
            for i in range(1, 4):
                (images_dir / f"{i}.jpg").touch()
                (masks_dir / f"{i}.png").touch()

        return FolderDataset(root=tmp_path, n_shots=1)

    def test_reference_filtering(self, multi_category_dataset: FolderDataset) -> None:
        """Test reference sample filtering."""
        ref_df = multi_category_dataset.get_reference_samples_df()
        # Should have 3 reference samples (1 per category)
        assert len(ref_df) == 3

        target_df = multi_category_dataset.get_target_samples_df()
        # Should have 6 target samples (2 per category)
        assert len(target_df) == 6

    def test_category_filtering(self, multi_category_dataset: FolderDataset) -> None:
        """Test category-based filtering."""
        apple_ref_df = multi_category_dataset.get_reference_samples_df(category="apple")
        assert len(apple_ref_df) == 1

        apple_target_df = multi_category_dataset.get_target_samples_df(category="apple")
        assert len(apple_target_df) == 2

    def test_get_reference_dataset(self, multi_category_dataset: FolderDataset) -> None:
        """Test getting reference dataset."""
        ref_dataset = multi_category_dataset.get_reference_dataset()
        assert len(ref_dataset) == 3
        # All should have at least one reference
        assert ref_dataset.df["is_reference"].list.contains(item=True).all()

    def test_get_target_dataset(self, multi_category_dataset: FolderDataset) -> None:
        """Test getting target dataset."""
        target_dataset = multi_category_dataset.get_target_dataset()
        assert len(target_dataset) == 6
        # All should have no reference instances
        assert not target_dataset.df["is_reference"].list.contains(item=True).any()


class TestFolderDatasetBatch:
    """Test FolderDataset batch creation."""

    @pytest.fixture
    def batch_dataset(self, tmp_path: Path) -> FolderDataset:
        """Create a dataset for batch testing."""
        images_dir = tmp_path / "images" / "test"
        masks_dir = tmp_path / "masks" / "test"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        for i in range(1, 4):
            (images_dir / f"{i}.jpg").touch()
            (masks_dir / f"{i}.png").touch()

        return FolderDataset(root=tmp_path, n_shots=1)

    @patch("instantlearn.data.base.base.read_image")
    @patch("instantlearn.data.folder.dataset.read_mask")
    def test_batch_creation(
        self,
        mock_read_mask: MagicMock,
        mock_read_image: MagicMock,
        batch_dataset: FolderDataset,
    ) -> None:
        """Test batch creation."""
        # Mock image and mask reading
        mock_read_image.return_value = torch.zeros((3, 100, 100))
        mock_read_mask.return_value = torch.zeros((100, 100), dtype=torch.uint8)

        samples = [batch_dataset[i] for i in range(len(batch_dataset))]
        batch = Batch.collate(samples)

        assert isinstance(batch, Batch)
        assert len(batch) == 3
        assert len(batch.categories) == 3
        assert len(batch.category_ids) == 3
        assert len(batch.is_reference) == 3

        # Each sample should have single instance
        assert all(len(cats) == 1 for cats in batch.categories)


class TestFolderDatasetEdgeCases:
    """Test FolderDataset edge cases."""

    def test_single_image_pair(self, tmp_path: Path) -> None:
        """Test dataset with single image-mask pair."""
        images_dir = tmp_path / "images" / "single"
        masks_dir = tmp_path / "masks" / "single"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)
        (images_dir / "1.jpg").touch()
        (masks_dir / "1.png").touch()

        dataset = FolderDataset(root=tmp_path, n_shots=1)
        assert len(dataset) == 1

    def test_n_shots_greater_than_samples(self, tmp_path: Path) -> None:
        """Test when n_shots is greater than available samples."""
        images_dir = tmp_path / "images" / "few"
        masks_dir = tmp_path / "masks" / "few"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        # Create only 2 pairs
        (images_dir / "1.jpg").touch()
        (masks_dir / "1.png").touch()
        (images_dir / "2.jpg").touch()
        (masks_dir / "2.png").touch()

        # Request 5 shots - should warn but still work
        dataset = FolderDataset(root=tmp_path, n_shots=5)
        assert len(dataset) == 2
        # All should be references since n_shots > available
        ref_df = dataset.get_reference_samples_df()

        ref_df = ref_df.filter(pl.col("categories").list.contains("few"))
        # 2 shots as there are only 2 samples even though n_shots is 5
        assert len(ref_df) == 2

    def test_different_file_extensions(self, tmp_path: Path) -> None:
        """Test with different file extensions."""
        images_dir = tmp_path / "images" / "ext"
        masks_dir = tmp_path / "masks" / "ext"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        # Create different extensions
        (images_dir / "1.jpeg").touch()
        (masks_dir / "1.bmp").touch()

        dataset = FolderDataset(
            root=tmp_path,
            img_extensions=(".jpg", ".jpeg"),
            mask_extensions=(".png", ".bmp"),
        )
        assert len(dataset) == 1

    def test_missing_mask_for_image(self, tmp_path: Path) -> None:
        """Test that images without matching masks are skipped."""
        images_dir = tmp_path / "images" / "partial"
        masks_dir = tmp_path / "masks" / "partial"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        # Create 3 images but only 2 masks
        (images_dir / "1.jpg").touch()
        (masks_dir / "1.png").touch()
        (images_dir / "2.jpg").touch()
        (masks_dir / "2.png").touch()
        (images_dir / "3.jpg").touch()  # No matching mask

        dataset = FolderDataset(root=tmp_path)
        assert len(dataset) == 2  # Only valid pairs

    def test_sorted_file_ordering(self, tmp_path: Path) -> None:
        """Test that files are sorted correctly."""
        images_dir = tmp_path / "images" / "sorted"
        masks_dir = tmp_path / "masks" / "sorted"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        # Create files with different naming patterns
        for i in [1, 2, 10, 20]:
            (images_dir / f"{i}.jpg").touch()
            (masks_dir / f"{i}.png").touch()

        dataset = FolderDataset(root=tmp_path, n_shots=2)

        # Check that first two samples are references (should be 1 and 2, not 10 and 20)
        sample0 = dataset.df.row(0, named=True)
        sample1 = dataset.df.row(1, named=True)

        # The first sample should be reference
        assert sample0["is_reference"][0] is True
        assert sample1["is_reference"][0] is True
