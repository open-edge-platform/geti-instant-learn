# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for PerSegDataset and GetiPromptDataset."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from getiprompt.data.datasets import PerSegDataset
from getiprompt.data.sample import GetiPromptSample
from getiprompt.data.batch import GetiPromptBatch


# Test configuration
PERSEG_ROOT = Path("/home/yuchunli/datasets/PerSeg")


@pytest.fixture
def perseg_dataset():
    """Create a PerSeg dataset with 1 shot."""
    return PerSegDataset(root=PERSEG_ROOT, n_shots=1)


@pytest.fixture
def perseg_dataset_2_shots():
    """Create a PerSeg dataset with 2 shots."""
    return PerSegDataset(root=PERSEG_ROOT, n_shots=2)


@pytest.fixture
def perseg_dataset_filtered():
    """Create a PerSeg dataset with filtered categories."""
    return PerSegDataset(
        root=PERSEG_ROOT,
        categories=["backpack", "dog", "cat"],
        n_shots=1
    )


class TestPerSegDatasetBasics:
    """Test basic dataset functionality."""
    
    def test_dataset_creation(self, perseg_dataset):
        """Test that dataset can be created."""
        assert perseg_dataset is not None
        assert isinstance(perseg_dataset, PerSegDataset)
    
    def test_dataset_length(self, perseg_dataset):
        """Test dataset length."""
        length = len(perseg_dataset)
        assert length > 0
        print(f"\nDataset has {length} samples")
    
    def test_dataset_name(self, perseg_dataset):
        """Test dataset name property."""
        assert perseg_dataset.name == "PerSeg"
    
    def test_categories(self, perseg_dataset):
        """Test categories property."""
        categories = perseg_dataset.categories
        assert isinstance(categories, list)
        assert len(categories) > 0
        assert all(isinstance(cat, str) for cat in categories)
        print(f"\nFound {len(categories)} categories: {categories[:5]}...")
    
    def test_category_ids(self, perseg_dataset):
        """Test category_ids property."""
        category_ids = perseg_dataset.category_ids
        assert isinstance(category_ids, list)
        assert len(category_ids) > 0
        assert all(isinstance(cid, int) for cid in category_ids)
        assert category_ids == sorted(category_ids)  # Should be sorted
    
    def test_num_categories(self, perseg_dataset):
        """Test num_categories property."""
        num_cats = perseg_dataset.num_categories
        assert num_cats == len(perseg_dataset.categories)
        print(f"\nNumber of categories: {num_cats}")


class TestPerSegDatasetSampling:
    """Test dataset sampling and item retrieval."""
    
    def test_getitem_single_sample(self, perseg_dataset):
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
        assert sample.image.shape[0] == 3  # CHW format
        print(f"\nSample image shape: {sample.image.shape}")
    
    def test_sample_has_mask(self, perseg_dataset):
        """Test that samples have masks."""
        sample = perseg_dataset[0]
        
        assert sample.masks is not None
        assert isinstance(sample.masks, np.ndarray)
        assert sample.masks.ndim == 2  # Single mask (H, W) for PerSeg
        print(f"\nSample mask shape: {sample.masks.shape}")
    
    def test_sample_metadata(self, perseg_dataset):
        """Test sample metadata fields."""
        sample = perseg_dataset[0]
        
        # Check metadata exists
        assert sample.categories is not None
        assert sample.category_ids is not None
        assert isinstance(sample.is_reference, bool)
        assert isinstance(sample.n_shot, int)
        
        print(f"\nSample metadata:")
        print(f"  Category: {sample.categories}")
        print(f"  Category ID: {sample.category_ids}")
        print(f"  Is reference: {sample.is_reference}")
        print(f"  N-shot: {sample.n_shot}")
    
    def test_multiple_samples(self, perseg_dataset):
        """Test retrieving multiple samples."""
        num_samples = min(5, len(perseg_dataset))
        
        for i in range(num_samples):
            sample = perseg_dataset[i]
            assert isinstance(sample, GetiPromptSample)
            assert sample.image.shape[0] == 3
    
    def test_reference_vs_target_samples(self, perseg_dataset):
        """Test that reference and target samples are correctly labeled."""
        # Get all samples
        all_samples = [perseg_dataset[i] for i in range(len(perseg_dataset))]
        
        # Check reference samples
        reference_samples = [s for s in all_samples if s.is_reference]
        target_samples = [s for s in all_samples if not s.is_reference]
        
        assert len(reference_samples) > 0
        assert len(target_samples) > 0
        
        # Check n_shot values
        for sample in reference_samples:
            assert sample.n_shot >= 0  # Reference samples have n_shot >= 0
        
        for sample in target_samples:
            assert sample.n_shot == -1  # Target samples have n_shot = -1
        
        print(f"\nReference samples: {len(reference_samples)}")
        print(f"Target samples: {len(target_samples)}")


class TestPerSegDatasetFiltering:
    """Test dataset filtering and subsetting."""
    
    def test_filtered_categories(self, perseg_dataset_filtered):
        """Test dataset with filtered categories."""
        categories = perseg_dataset_filtered.categories
        
        assert len(categories) == 3
        assert "backpack" in categories
        assert "dog" in categories
        assert "cat" in categories
    
    def test_get_reference_dataset(self, perseg_dataset):
        """Test getting reference samples only."""
        ref_dataset = perseg_dataset.get_reference_dataset()
        
        assert len(ref_dataset) > 0
        assert len(ref_dataset) <= len(perseg_dataset)
        
        # Check all samples are reference samples
        for i in range(len(ref_dataset)):
            sample = ref_dataset[i]
            assert sample.is_reference is True
            assert sample.n_shot >= 0
        
        print(f"\nReference dataset size: {len(ref_dataset)}")
    
    def test_get_target_dataset(self, perseg_dataset):
        """Test getting target samples only."""
        target_dataset = perseg_dataset.get_target_dataset()
        
        assert len(target_dataset) > 0
        assert len(target_dataset) <= len(perseg_dataset)
        
        # Check all samples are target samples
        for i in range(min(5, len(target_dataset))):
            sample = target_dataset[i]
            assert sample.is_reference is False
            assert sample.n_shot == -1
        
        print(f"\nTarget dataset size: {len(target_dataset)}")
    
    def test_get_reference_dataset_by_category(self, perseg_dataset):
        """Test getting reference samples for a specific category."""
        category = perseg_dataset.categories[0]
        ref_dataset = perseg_dataset.get_reference_dataset(category=category)
        
        assert len(ref_dataset) > 0
        
        # Check all samples are reference samples of the correct category
        for i in range(len(ref_dataset)):
            sample = ref_dataset[i]
            assert sample.is_reference is True
            assert category in sample.categories
        
        print(f"\nReference samples for '{category}': {len(ref_dataset)}")
    
    def test_subsample(self, perseg_dataset):
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
    
    def test_dataset_concatenation(self, perseg_dataset):
        """Test concatenating datasets."""
        # Get two subsets
        subset1 = perseg_dataset.subsample([0, 1, 2])
        subset2 = perseg_dataset.subsample([3, 4, 5])
        
        # Concatenate
        combined = subset1 + subset2
        
        assert len(combined) == len(subset1) + len(subset2)


class TestPerSegDatasetNShots:
    """Test n_shots functionality."""
    
    def test_n_shots_1(self, perseg_dataset):
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
    
    def test_n_shots_2(self, perseg_dataset_2_shots):
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
        
        print(f"\nCategory reference counts (2 shots): {category_counts}")


class TestPerSegDatasetCategoryMapping:
    """Test category name/ID mapping."""
    
    def test_get_category_name(self, perseg_dataset):
        """Test getting category name from ID."""
        category_id = perseg_dataset.category_ids[0]
        category_name = perseg_dataset.get_category_name(category_id)
        
        assert isinstance(category_name, str)
        assert category_name in perseg_dataset.categories
        print(f"\nCategory ID {category_id} -> {category_name}")
    
    def test_get_category_id(self, perseg_dataset):
        """Test getting category ID from name."""
        category_name = perseg_dataset.categories[0]
        category_id = perseg_dataset.get_category_id(category_name)
        
        assert isinstance(category_id, int)
        assert category_id in perseg_dataset.category_ids
        print(f"\nCategory '{category_name}' -> ID {category_id}")
    
    def test_category_roundtrip(self, perseg_dataset):
        """Test category name <-> ID roundtrip conversion."""
        original_name = perseg_dataset.categories[0]
        
        # Name -> ID -> Name
        category_id = perseg_dataset.get_category_id(original_name)
        recovered_name = perseg_dataset.get_category_name(category_id)
        
        assert original_name == recovered_name
    
    def test_invalid_category_id(self, perseg_dataset):
        """Test error handling for invalid category ID."""
        with pytest.raises(ValueError, match="Category ID .* not found"):
            perseg_dataset.get_category_name(99999)
    
    def test_invalid_category_name(self, perseg_dataset):
        """Test error handling for invalid category name."""
        with pytest.raises(ValueError, match="Category .* not found"):
            perseg_dataset.get_category_id("nonexistent_category")


class TestPerSegDatasetDataLoader:
    """Test PyTorch DataLoader integration."""
    
    def test_dataloader_creation(self, perseg_dataset):
        """Test creating a DataLoader."""
        dataloader = DataLoader(
            perseg_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=perseg_dataset.collate_fn
        )
        
        assert dataloader is not None
    
    def test_dataloader_iteration(self, perseg_dataset):
        """Test iterating through DataLoader."""
        dataloader = DataLoader(
            perseg_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=perseg_dataset.collate_fn
        )
        
        # Get first batch
        batch = next(iter(dataloader))
        
        # Check batch type
        assert isinstance(batch, GetiPromptBatch)
        
        # Check batch fields
        assert hasattr(batch, "images")
        assert hasattr(batch, "masks")
        assert hasattr(batch, "categories")
        
        print(f"\nBatch images shape: {batch.images.shape if hasattr(batch.images, 'shape') else 'N/A'}")
        print(f"Batch size: {len(batch.images) if isinstance(batch.images, list) else batch.images.shape[0]}")
    
    def test_dataloader_multiple_batches(self, perseg_dataset):
        """Test iterating through multiple batches."""
        batch_size = 4
        dataloader = DataLoader(
            perseg_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=perseg_dataset.collate_fn
        )
        
        num_batches = 3
        batches = []
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            batches.append(batch)
            assert isinstance(batch, GetiPromptBatch)
        
        assert len(batches) == num_batches
        print(f"\nProcessed {num_batches} batches successfully")


class TestPerSegDatasetDataFrame:
    """Test underlying Polars DataFrame."""
    
    def test_df_property(self, perseg_dataset):
        """Test accessing the DataFrame property."""
        df = perseg_dataset.df
        
        assert df is not None
        assert len(df) == len(perseg_dataset)
        
        # Check required columns
        required_cols = {"category", "category_id", "image_path", "mask_path", "is_reference", "n_shot"}
        assert required_cols.issubset(set(df.columns))
        
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {df.columns}")
    
    def test_get_reference_samples_df(self, perseg_dataset):
        """Test getting reference samples DataFrame."""
        ref_df = perseg_dataset.get_reference_samples_df()
        
        assert ref_df is not None
        assert len(ref_df) > 0
        
        # Check all are reference samples
        assert ref_df["is_reference"].all()
        
        print(f"\nReference DataFrame shape: {ref_df.shape}")
    
    def test_get_target_samples_df(self, perseg_dataset):
        """Test getting target samples DataFrame."""
        target_df = perseg_dataset.get_target_samples_df()
        
        assert target_df is not None
        assert len(target_df) > 0
        
        # Check all are target samples
        assert not target_df["is_reference"].any()
        
        print(f"\nTarget DataFrame shape: {target_df.shape}")


class TestPerSegDatasetPaths:
    """Test file path handling."""
    
    def test_image_paths_exist(self, perseg_dataset):
        """Test that image paths exist."""
        num_samples_to_check = min(5, len(perseg_dataset))
        
        for i in range(num_samples_to_check):
            sample = perseg_dataset[i]
            image_path = Path(sample.image_path)
            assert image_path.exists(), f"Image not found: {image_path}"
    
    def test_mask_paths_exist(self, perseg_dataset):
        """Test that mask paths exist."""
        # Check from DataFrame
        df = perseg_dataset.df
        num_samples_to_check = min(5, len(df))
        
        for i in range(num_samples_to_check):
            mask_path = Path(df["mask_path"][i])
            assert mask_path.exists(), f"Mask not found: {mask_path}"
    
    def test_paths_match_category(self, perseg_dataset):
        """Test that paths contain the correct category name."""
        sample = perseg_dataset[0]
        category = sample.categories[0] if isinstance(sample.categories, list) else sample.categories
        
        # Check image path contains category
        assert category in str(sample.image_path)


class TestPerSegDatasetEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_root_path(self):
        """Test error handling for invalid root path."""
        with pytest.raises(FileNotFoundError):
            PerSegDataset(root="/nonexistent/path")
    
    def test_invalid_categories(self):
        """Test error handling for invalid categories."""
        with pytest.raises(ValueError):
            PerSegDataset(
                root=PERSEG_ROOT,
                categories=["nonexistent_category"]
            )
    
    def test_invalid_index(self, perseg_dataset):
        """Test error handling for invalid index."""
        with pytest.raises((IndexError, Exception)):
            _ = perseg_dataset[len(perseg_dataset) + 100]
    
    def test_negative_index(self, perseg_dataset):
        """Test negative indexing."""
        sample = perseg_dataset[-1]
        assert isinstance(sample, GetiPromptSample)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

