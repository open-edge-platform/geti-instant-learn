# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for PerSegDataset and GetiPromptDataset."""

from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import DataLoader

from getiprompt.data.batch import GetiPromptBatch
from getiprompt.data.datasets import PerSegDataset
from getiprompt.data.sample import GetiPromptSample

# Test configuration
PERSEG_ROOT = Path("/home/yuchunli/datasets/PerSeg")


@pytest.fixture
def perseg_dataset() -> PerSegDataset:
    """Create a PerSeg dataset with 1 shot."""
    return PerSegDataset(root=PERSEG_ROOT, n_shots=1)


@pytest.fixture
def perseg_dataset_2_shots() -> PerSegDataset:
    """Create a PerSeg dataset with 2 shots."""
    return PerSegDataset(root=PERSEG_ROOT, n_shots=2)


@pytest.fixture
def perseg_dataset_filtered() -> PerSegDataset:
    """Create a PerSeg dataset with filtered categories."""
    return PerSegDataset(
        root=PERSEG_ROOT,
        categories=["backpack", "dog", "cat"],
        n_shots=1,
    )


class TestPerSegDatasetBasics:
    """Test basic dataset functionality."""

    def test_dataset_creation(self, perseg_dataset: PerSegDataset) -> None:
        """Test that dataset can be created."""
        assert perseg_dataset is not None
        assert isinstance(perseg_dataset, PerSegDataset)

    def test_dataset_length(self, perseg_dataset: PerSegDataset) -> None:
        """Test dataset length."""
        length = len(perseg_dataset)
        assert length > 0

    def test_dataset_name(self, perseg_dataset: PerSegDataset) -> None:
        """Test dataset name property."""
        assert perseg_dataset.name == "PerSeg"

    def test_categories(self, perseg_dataset: PerSegDataset) -> None:
        """Test categories property."""
        categories = perseg_dataset.categories
        assert isinstance(categories, list)
        assert len(categories) > 0
        assert all(isinstance(cat, str) for cat in categories)

    def test_category_ids(self, perseg_dataset: PerSegDataset) -> None:
        """Test category_ids property."""
        category_ids = perseg_dataset.category_ids
        assert isinstance(category_ids, list)
        assert len(category_ids) > 0
        assert all(isinstance(cid, int) for cid in category_ids)
        assert category_ids == sorted(category_ids)  # Should be sorted

    def test_num_categories(self, perseg_dataset: PerSegDataset) -> None:
        """Test num_categories property."""
        num_cats = perseg_dataset.num_categories
        assert num_cats == len(perseg_dataset.categories)


class TestPerSegDatasetSampling:
    """Test dataset sampling and item retrieval."""

    def test_getitem_single_sample(self, perseg_dataset: PerSegDataset) -> None:
        """Test retrieving a single sample."""
        sample = perseg_dataset[0]

        # Check type
        assert isinstance(sample, GetiPromptSample)

        # Check required fields
        assert sample.image is not None
        assert sample.image_path is not None

        # Check image properties
        assert isinstance(sample.image, np.ndarray)
        assert sample.image.ndim == 3
        assert sample.image.shape[2] == 3  # HWC format (H, W, C)

    def test_sample_has_mask(self, perseg_dataset: PerSegDataset) -> None:
        """Test that samples have masks."""
        sample = perseg_dataset[0]

        assert sample.masks is not None
        assert isinstance(sample.masks, np.ndarray)
        assert sample.masks.ndim == 3  # Multi-instance format (N, H, W)
        assert sample.masks.shape[0] == 1  # PerSeg has 1 instance

    def test_sample_metadata(self, perseg_dataset: PerSegDataset) -> None:
        """Test sample metadata fields."""
        sample = perseg_dataset[0]

        # Check metadata exists
        assert sample.categories is not None
        assert sample.category_ids is not None
        # PerSeg is single-instance, so lists have length 1
        assert isinstance(sample.is_reference, list)
        assert len(sample.is_reference) == 1
        assert isinstance(sample.is_reference[0], bool)
        assert isinstance(sample.n_shot, list)
        assert len(sample.n_shot) == 1
        assert isinstance(sample.n_shot[0], int)

    def test_multiple_samples(self, perseg_dataset: PerSegDataset) -> None:
        """Test retrieving multiple samples."""
        num_samples = min(5, len(perseg_dataset))

        for i in range(num_samples):
            sample = perseg_dataset[i]
            assert isinstance(sample, GetiPromptSample)
            assert sample.image.shape[2] == 3  # HWC format

    def test_reference_vs_target_samples(self, perseg_dataset: PerSegDataset) -> None:
        """Test that reference and target samples are correctly labeled."""
        # Get all samples
        all_samples = [perseg_dataset[i] for i in range(len(perseg_dataset))]

        # Check reference samples (for single-instance, check first element)
        reference_samples = [s for s in all_samples if s.is_reference[0]]
        target_samples = [s for s in all_samples if not s.is_reference[0]]

        assert len(reference_samples) > 0
        assert len(target_samples) > 0

        # Check n_shot values
        for sample in reference_samples:
            assert sample.n_shot[0] >= 0  # Reference samples have n_shot >= 0

        for sample in target_samples:
            assert sample.n_shot[0] == -1  # Target samples have n_shot = -1


class TestPerSegDatasetFiltering:
    """Test dataset filtering and subsetting."""

    def test_filtered_categories(self, perseg_dataset_filtered: PerSegDataset) -> None:
        """Test dataset with filtered categories."""
        categories = perseg_dataset_filtered.categories

        assert len(categories) == 3
        assert "backpack" in categories
        assert "dog" in categories
        assert "cat" in categories

    def test_get_reference_dataset(self, perseg_dataset: PerSegDataset) -> None:
        """Test getting reference samples only."""
        ref_dataset = perseg_dataset.get_reference_dataset()

        assert len(ref_dataset) > 0
        assert len(ref_dataset) <= len(perseg_dataset)

        # Check all samples are reference samples
        for i in range(len(ref_dataset)):
            sample = ref_dataset[i]
            assert any(sample.is_reference)  # At least one instance is reference
            assert sample.n_shot[0] >= 0  # Check first instance

    def test_get_target_dataset(self, perseg_dataset: PerSegDataset) -> None:
        """Test getting target samples only."""
        target_dataset = perseg_dataset.get_target_dataset()

        assert len(target_dataset) > 0
        assert len(target_dataset) <= len(perseg_dataset)

        # Check all samples are target samples
        for i in range(min(5, len(target_dataset))):
            sample = target_dataset[i]
            assert not any(sample.is_reference)  # No instance is reference
            assert sample.n_shot[0] == -1  # Check first instance

    def test_get_reference_dataset_by_category(self, perseg_dataset: PerSegDataset) -> None:
        """Test getting reference samples for a specific category."""
        category = perseg_dataset.categories[0]
        ref_dataset = perseg_dataset.get_reference_dataset(category=category)

        assert len(ref_dataset) > 0

        # Check all samples are reference samples of the correct category
        for i in range(len(ref_dataset)):
            sample = ref_dataset[i]
            assert any(sample.is_reference)  # At least one instance is reference
            assert category in sample.categories

    def test_subsample(self, perseg_dataset: PerSegDataset) -> None:
        """Test subsampling the dataset."""
        # Get a subset of indices
        indices = [0, 2, 4, 6, 8]
        subset = perseg_dataset.subsample(indices)

        assert len(subset) == len(indices)

        # Check samples match
        for i, idx in enumerate(indices):
            original_sample = perseg_dataset[idx]
            subset_sample = subset[i]
            assert original_sample.image_path == subset_sample.image_path

    def test_dataset_concatenation(self, perseg_dataset: PerSegDataset) -> None:
        """Test concatenating datasets."""
        # Get two subsets
        subset1 = perseg_dataset.subsample([0, 1, 2])
        subset2 = perseg_dataset.subsample([3, 4, 5])

        # Concatenate
        combined = subset1 + subset2

        assert len(combined) == len(subset1) + len(subset2)


class TestPerSegDatasetNShots:
    """Test n_shots functionality."""

    def test_n_shots_1(self, perseg_dataset: PerSegDataset) -> None:
        """Test with 1 reference shot per category."""
        ref_dataset = perseg_dataset.get_reference_dataset()

        # Count reference samples per category
        category_counts = {}
        for i in range(len(ref_dataset)):
            sample = ref_dataset[i]
            cat = sample.categories[0] if isinstance(sample.categories, list) else sample.categories
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Check each category has 1 reference sample
        for cat, count in category_counts.items():
            assert count == 1, f"Category {cat} has {count} reference samples, expected 1"

    def test_n_shots_2(self, perseg_dataset_2_shots: PerSegDataset) -> None:
        """Test with 2 reference shots per category."""
        ref_dataset = perseg_dataset_2_shots.get_reference_dataset()

        # Count reference samples per category
        category_counts = {}
        for i in range(len(ref_dataset)):
            sample = ref_dataset[i]
            cat = sample.categories[0] if isinstance(sample.categories, list) else sample.categories
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Check each category has up to 2 reference samples
        for cat, count in category_counts.items():
            assert count <= 2, f"Category {cat} has {count} reference samples, expected <= 2"


class TestPerSegDatasetCategoryMapping:
    """Test category name/ID mapping."""

    def test_get_category_name(self, perseg_dataset: PerSegDataset) -> None:
        """Test getting category name from ID."""
        category_id = perseg_dataset.category_ids[0]
        category_name = perseg_dataset.get_category_name(category_id)

        assert isinstance(category_name, str)
        assert category_name in perseg_dataset.categories

    def test_get_category_id(self, perseg_dataset: PerSegDataset) -> None:
        """Test getting category ID from name."""
        category_name = perseg_dataset.categories[0]
        category_id = perseg_dataset.get_category_id(category_name)

        assert isinstance(category_id, int)
        assert category_id in perseg_dataset.category_ids

    def test_category_roundtrip(self, perseg_dataset: PerSegDataset) -> None:
        """Test category name <-> ID roundtrip conversion."""
        original_name = perseg_dataset.categories[0]

        # Name -> ID -> Name
        category_id = perseg_dataset.get_category_id(original_name)
        recovered_name = perseg_dataset.get_category_name(category_id)

        assert original_name == recovered_name

    def test_invalid_category_id(self, perseg_dataset: PerSegDataset) -> None:
        """Test error handling for invalid category ID."""
        with pytest.raises(ValueError, match=r"Category ID .* not found"):
            perseg_dataset.get_category_name(99999)

    def test_invalid_category_name(self, perseg_dataset: PerSegDataset) -> None:
        """Test error handling for invalid category name."""
        with pytest.raises(ValueError, match=r"Category .* not found"):
            perseg_dataset.get_category_id("nonexistent_category")


class TestPerSegDatasetDataLoader:
    """Test PyTorch DataLoader integration."""

    def test_dataloader_creation(self, perseg_dataset: PerSegDataset) -> None:
        """Test creating a DataLoader."""
        dataloader = DataLoader(
            perseg_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=perseg_dataset.collate_fn,
        )

        assert dataloader is not None

    def test_dataloader_iteration(self, perseg_dataset: PerSegDataset) -> None:
        """Test iterating through DataLoader."""
        dataloader = DataLoader(
            perseg_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=perseg_dataset.collate_fn,
        )

        # Get first batch
        batch = next(iter(dataloader))

        # Check batch type
        assert isinstance(batch, GetiPromptBatch)

        # Check batch fields
        assert hasattr(batch, "images")
        assert hasattr(batch, "masks")
        assert hasattr(batch, "categories")

    def test_dataloader_multiple_batches(self, perseg_dataset: PerSegDataset) -> None:
        """Test iterating through multiple batches."""
        batch_size = 4
        dataloader = DataLoader(
            perseg_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=perseg_dataset.collate_fn,
        )

        num_batches = 3
        batches = []

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            batches.append(batch)
            assert isinstance(batch, GetiPromptBatch)

        assert len(batches) == num_batches


class TestPerSegDatasetDataFrame:
    """Test underlying Polars DataFrame."""

    def test_df_property(self, perseg_dataset: PerSegDataset) -> None:
        """Test accessing the DataFrame property."""
        dataframe = perseg_dataset.df

        assert dataframe is not None
        assert len(dataframe) == len(perseg_dataset)

        # Check required columns (updated for multi-instance schema)
        required_cols = {"categories", "category_ids", "image_path", "mask_paths", "is_reference", "n_shot"}
        assert required_cols.issubset(set(dataframe.columns))

    def test_get_reference_samples_df(self, perseg_dataset: PerSegDataset) -> None:
        """Test getting reference samples DataFrame."""
        ref_df = perseg_dataset.get_reference_samples_df()

        assert ref_df is not None
        assert len(ref_df) > 0

        # Check all are reference samples (is_reference is list[bool])
        assert all(any(row) for row in ref_df["is_reference"])

    def test_get_target_samples_df(self, perseg_dataset: PerSegDataset) -> None:
        """Test getting target samples DataFrame."""
        target_df = perseg_dataset.get_target_samples_df()

        assert target_df is not None
        assert len(target_df) > 0

        # Check all are target samples (is_reference is list[bool])
        assert all(not any(row) for row in target_df["is_reference"])


class TestPerSegDatasetPaths:
    """Test file path handling."""

    def test_image_paths_exist(self, perseg_dataset: PerSegDataset) -> None:
        """Test that image paths exist."""
        num_samples_to_check = min(5, len(perseg_dataset))

        for i in range(num_samples_to_check):
            sample = perseg_dataset[i]
            image_path = Path(sample.image_path)
            assert image_path.exists(), f"Image not found: {image_path}"

    def test_mask_paths_exist(self, perseg_dataset: PerSegDataset) -> None:
        """Test that mask paths exist."""
        # Check from DataFrame
        dataframe = perseg_dataset.df
        num_samples_to_check = min(5, len(dataframe))

        for i in range(num_samples_to_check):
            mask_paths = dataframe["mask_paths"][i]  # Get list of paths
            mask_path = Path(mask_paths[0])  # Get first path
            assert mask_path.exists(), f"Mask not found: {mask_path}"

    def test_paths_match_category(self, perseg_dataset: PerSegDataset) -> None:
        """Test that paths contain the correct category name."""
        sample = perseg_dataset[0]
        category = sample.categories[0] if isinstance(sample.categories, list) else sample.categories

        # Check image path contains category
        assert category in str(sample.image_path)


class TestPerSegDatasetEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_root_path(self) -> None:
        """Test error handling for invalid root path."""
        with pytest.raises(FileNotFoundError):
            PerSegDataset(root="/nonexistent/path")

    def test_invalid_categories(self) -> None:
        """Test error handling for invalid categories."""
        with pytest.raises(ValueError, match=r"No valid categories found"):
            PerSegDataset(
                root=PERSEG_ROOT,
                categories=["nonexistent_category"],
            )

    def test_invalid_index(self, perseg_dataset: PerSegDataset) -> None:
        """Test error handling for invalid index."""
        with pytest.raises((IndexError, Exception)):
            _ = perseg_dataset[len(perseg_dataset) + 100]

    def test_negative_index(self, perseg_dataset: PerSegDataset) -> None:
        """Test negative indexing."""
        sample = perseg_dataset[-1]
        assert isinstance(sample, GetiPromptSample)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
