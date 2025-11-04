# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Batch class."""

import numpy as np
import pytest
import torch

from getiprompt.data.base.batch import Batch
from getiprompt.data.base.sample import Sample

# Create a random generator for consistent testing
_rng = np.random.default_rng(42)


@pytest.fixture
def sample_single_instance() -> Sample:
    """Create a single-instance sample."""
    return Sample(
        image=_rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        image_path="single.jpg",
        masks=_rng.integers(0, 2, (1, 224, 224), dtype=np.uint8),
        bboxes=np.array([[10, 20, 100, 120]], dtype=np.float32),
        points=np.array([[50, 60]], dtype=np.float32),
        categories=["cat"],
        category_ids=np.array([0], dtype=np.int32),
        mask_paths=["mask1.png"],
        is_reference=[True],
        n_shot=[0],
    )


@pytest.fixture
def sample_multi_instance() -> Sample:
    """Create a multi-instance sample."""
    return Sample(
        image=_rng.integers(0, 255, (512, 512, 3), dtype=np.uint8),
        image_path="multi.jpg",
        masks=_rng.integers(0, 2, (3, 512, 512), dtype=np.uint8),
        bboxes=np.array([[10, 20, 110, 120], [200, 150, 350, 270], [100, 100, 200, 200]], dtype=np.float32),
        points=np.array([[50, 60], [250, 200], [150, 150]], dtype=np.float32),
        categories=["person", "car", "dog"],
        category_ids=np.array([0, 1, 2], dtype=np.int32),
        mask_paths=["mask1.png", "mask2.png", "mask3.png"],
        is_reference=[True, False, True],
        n_shot=[0, -1, 1],
    )


@pytest.fixture
def sample_no_masks() -> Sample:
    """Create a sample without masks."""
    return Sample(
        image=_rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        image_path="no_masks.jpg",
        categories=["cat"],
        category_ids=np.array([0], dtype=np.int32),
        is_reference=[True],
        n_shot=[0],
    )


class TestBatchBasic:
    """Test Batch basic functionality."""

    def test_batch_creation(self, sample_single_instance: Sample, sample_multi_instance: Sample) -> None:
        """Test batch creation from samples."""
        samples = [sample_single_instance, sample_multi_instance]
        batch = Batch.collate(samples)

        assert len(batch) == 2
        assert batch.samples == samples

    def test_batch_direct_constructor(self, sample_single_instance: Sample, sample_multi_instance: Sample) -> None:
        """Test batch creation using direct constructor."""
        samples = [sample_single_instance, sample_multi_instance]
        batch = Batch(samples=samples)

        assert len(batch) == 2
        assert batch.samples == samples

    def test_batch_collate_empty_list(self) -> None:
        """Test that collating empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot collate empty list of samples"):
            Batch.collate([])

    def test_batch_length(self, sample_single_instance: Sample, sample_multi_instance: Sample) -> None:
        """Test batch length property."""
        samples = [sample_single_instance, sample_multi_instance]
        batch = Batch.collate(samples)

        assert len(batch) == 2

    def test_batch_indexing(self, sample_single_instance: Sample, sample_multi_instance: Sample) -> None:
        """Test batch indexing."""
        samples = [sample_single_instance, sample_multi_instance]
        batch = Batch.collate(samples)

        assert batch[0] == sample_single_instance
        assert batch[1] == sample_multi_instance

    def test_batch_iteration(self, sample_single_instance: Sample, sample_multi_instance: Sample) -> None:
        """Test batch iteration."""
        samples = [sample_single_instance, sample_multi_instance]
        batch = Batch.collate(samples)

        batch_samples = list(batch)
        assert batch_samples == samples

    def test_batch_images_property(self, sample_single_instance: Sample, sample_multi_instance: Sample) -> None:
        """Test batch images property."""
        samples = [sample_single_instance, sample_multi_instance]
        batch = Batch.collate(samples)

        images = batch.images
        assert len(images) == 2
        assert all(isinstance(img, torch.Tensor) for img in images)

        # Test caching
        images_cached = batch.images
        assert images is images_cached  # Same object due to caching

    def test_batch_masks_property(
        self,
        sample_single_instance: Sample,
        sample_multi_instance: Sample,
        sample_no_masks: Sample,
    ) -> None:
        """Test batch masks property."""
        samples = [sample_single_instance, sample_multi_instance, sample_no_masks]
        batch = Batch.collate(samples)

        masks = batch.masks
        assert len(masks) == 3
        assert isinstance(masks[0], torch.Tensor)  # Has masks
        assert isinstance(masks[1], torch.Tensor)  # Has masks
        assert masks[2] is None  # No masks

        # Test caching
        masks_cached = batch.masks
        assert masks is masks_cached  # Same object due to caching

    def test_batch_bboxes_property(
        self,
        sample_single_instance: Sample,
        sample_multi_instance: Sample,
        sample_no_masks: Sample,
    ) -> None:
        """Test batch bboxes property."""
        samples = [sample_single_instance, sample_multi_instance, sample_no_masks]
        batch = Batch.collate(samples)

        bboxes = batch.bboxes
        assert len(bboxes) == 3
        assert isinstance(bboxes[0], torch.Tensor)  # Has bboxes
        assert isinstance(bboxes[1], torch.Tensor)  # Has bboxes
        assert bboxes[2] is None  # No bboxes (sample_no_masks doesn't have bboxes)

    def test_batch_points_property(self, sample_single_instance: Sample, sample_multi_instance: Sample) -> None:
        """Test batch points property."""
        samples = [sample_single_instance, sample_multi_instance]
        batch = Batch.collate(samples)

        points = batch.points
        assert len(points) == 2
        assert all(isinstance(pts, torch.Tensor) for pts in points)

    def test_batch_categories_property(self, sample_single_instance: Sample, sample_multi_instance: Sample) -> None:
        """Test batch categories property."""
        samples = [sample_single_instance, sample_multi_instance]
        batch = Batch.collate(samples)

        categories = batch.categories
        assert len(categories) == 2
        assert categories[0] == ["cat"]  # Single instance
        assert categories[1] == ["person", "car", "dog"]  # Multi instance

    def test_batch_category_ids_property(self, sample_single_instance: Sample, sample_multi_instance: Sample) -> None:
        """Test batch category_ids property."""
        samples = [sample_single_instance, sample_multi_instance]
        batch = Batch.collate(samples)

        category_ids = batch.category_ids
        assert len(category_ids) == 2
        assert all(isinstance(ids, torch.Tensor) for ids in category_ids)
        assert category_ids[0].tolist() == [0]  # Single instance
        assert category_ids[1].tolist() == [0, 1, 2]  # Multi instance

    def test_batch_is_reference_property(self, sample_single_instance: Sample, sample_multi_instance: Sample) -> None:
        """Test batch is_reference property."""
        samples = [sample_single_instance, sample_multi_instance]
        batch = Batch.collate(samples)

        is_reference = batch.is_reference
        assert len(is_reference) == 2
        assert is_reference[0] == [True]  # Single instance
        assert is_reference[1] == [True, False, True]  # Multi instance

    def test_batch_n_shot_property(self, sample_single_instance: Sample, sample_multi_instance: Sample) -> None:
        """Test batch n_shot property."""
        samples = [sample_single_instance, sample_multi_instance]
        batch = Batch.collate(samples)

        n_shot = batch.n_shot
        assert len(n_shot) == 2
        assert n_shot[0] == [0]  # Single instance
        assert n_shot[1] == [0, -1, 1]  # Multi instance

    def test_batch_image_paths_property(self, sample_single_instance: Sample, sample_multi_instance: Sample) -> None:
        """Test batch image_paths property."""
        samples = [sample_single_instance, sample_multi_instance]
        batch = Batch.collate(samples)

        image_paths = batch.image_paths
        assert len(image_paths) == 2
        assert image_paths[0] == "single.jpg"
        assert image_paths[1] == "multi.jpg"

    def test_batch_mask_paths_property(
        self,
        sample_single_instance: Sample,
        sample_multi_instance: Sample,
        sample_no_masks: Sample,
    ) -> None:
        """Test batch mask_paths property."""
        samples = [sample_single_instance, sample_multi_instance, sample_no_masks]
        batch = Batch.collate(samples)

        mask_paths = batch.mask_paths
        assert len(mask_paths) == 3
        assert mask_paths[0] == ["mask1.png"]  # Single instance
        assert mask_paths[1] == ["mask1.png", "mask2.png", "mask3.png"]  # Multi instance
        assert mask_paths[2] is None  # No mask paths

    def test_batch_tensor_conversion_numpy_to_torch(self, sample_single_instance: Sample) -> None:
        """Test that numpy arrays are converted to torch tensors."""
        batch = Batch.collate([sample_single_instance])

        # Test image conversion
        images = batch.images
        assert isinstance(images[0], torch.Tensor)
        assert images[0].shape == (224, 224, 3)  # HWC format (as stored in Sample)

        # Test mask conversion
        masks = batch.masks
        assert isinstance(masks[0], torch.Tensor)
        assert masks[0].shape == (1, 224, 224)

        # Test bbox conversion
        bboxes = batch.bboxes
        assert isinstance(bboxes[0], torch.Tensor)
        assert bboxes[0].shape == (1, 4)

        # Test points conversion
        points = batch.points
        assert isinstance(points[0], torch.Tensor)
        assert points[0].shape == (1, 2)

        # Test category_ids conversion
        category_ids = batch.category_ids
        assert isinstance(category_ids[0], torch.Tensor)
        assert category_ids[0].shape == (1,)


class TestBatchTensorConversion:
    """Test Batch tensor conversion functionality."""

    def test_batch_tensor_conversion_torch_preserved(self) -> None:
        """Test that torch tensors are preserved without conversion."""
        # Create sample with torch tensors
        sample = Sample(
            image=torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8),
            image_path="test.jpg",
            masks=torch.randint(0, 2, (1, 224, 224), dtype=torch.uint8),
            bboxes=torch.tensor([[10, 20, 100, 120]], dtype=torch.float32),
            points=torch.tensor([[50, 60]], dtype=torch.float32),
            categories=["cat"],
            category_ids=torch.tensor([0], dtype=torch.int32),
            is_reference=[True],
            n_shot=[0],
        )

        batch = Batch.collate([sample])

        # Test that tensors are preserved
        images = batch.images
        assert images[0] is sample.image  # Same object

        masks = batch.masks
        assert masks[0] is sample.masks  # Same object

        bboxes = batch.bboxes
        assert bboxes[0] is sample.bboxes  # Same object

        points = batch.points
        assert points[0] is sample.points  # Same object

        category_ids = batch.category_ids
        assert category_ids[0] is sample.category_ids  # Same object

    def test_batch_mixed_numpy_torch(self) -> None:
        """Test batch with mixed numpy and torch tensors."""
        # Create samples with different tensor types
        sample_numpy = Sample(
            image=_rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
            image_path="numpy.jpg",
            masks=_rng.integers(0, 2, (1, 224, 224), dtype=np.uint8),
            categories=["cat"],
            category_ids=np.array([0], dtype=np.int32),
            is_reference=[True],
            n_shot=[0],
        )

        sample_torch = Sample(
            image=torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8),
            image_path="torch.jpg",
            masks=torch.randint(0, 2, (1, 224, 224), dtype=torch.uint8),
            categories=["dog"],
            category_ids=torch.tensor([1], dtype=torch.int32),
            is_reference=[False],
            n_shot=[-1],
        )

        batch = Batch.collate([sample_numpy, sample_torch])

        # Test that numpy arrays are converted and torch tensors are preserved
        images = batch.images
        assert isinstance(images[0], torch.Tensor)  # Converted from numpy
        assert isinstance(images[1], torch.Tensor)  # Already torch

        masks = batch.masks
        assert isinstance(masks[0], torch.Tensor)  # Converted from numpy
        assert isinstance(masks[1], torch.Tensor)  # Already torch

        category_ids = batch.category_ids
        assert isinstance(category_ids[0], torch.Tensor)  # Converted from numpy
        assert isinstance(category_ids[1], torch.Tensor)  # Already torch

    def test_batch_empty_tensors(self) -> None:
        """Test batch with empty tensors."""
        sample = Sample(
            image=_rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
            image_path="empty.jpg",
            categories=[],
            category_ids=np.array([], dtype=np.int32),
            is_reference=[],
            n_shot=[],
        )

        batch = Batch.collate([sample])

        # Test empty category_ids
        category_ids = batch.category_ids
        assert isinstance(category_ids[0], torch.Tensor)
        assert category_ids[0].shape == (0,)

        # Test empty categories
        categories = batch.categories
        assert categories[0] == []

    def test_batch_lazy_conversion(self, sample_single_instance: Sample) -> None:
        """Test that tensor conversion is lazy."""
        batch = Batch.collate([sample_single_instance])

        # Access images property multiple times
        images1 = batch.images
        images2 = batch.images

        # Should be the same object due to caching
        assert images1 is images2

        # Access masks property multiple times
        masks1 = batch.masks
        masks2 = batch.masks

        # Should be the same object due to caching
        assert masks1 is masks2

    def test_batch_property_independence(self, sample_single_instance: Sample, sample_multi_instance: Sample) -> None:
        """Test that different properties are independent."""
        batch = Batch.collate([sample_single_instance, sample_multi_instance])

        # Access different properties
        images = batch.images
        masks = batch.masks
        bboxes = batch.bboxes
        points = batch.points
        categories = batch.categories
        category_ids = batch.category_ids
        is_reference = batch.is_reference
        n_shot = batch.n_shot
        image_paths = batch.image_paths
        mask_paths = batch.mask_paths

        # All should work independently
        assert len(images) == 2
        assert len(masks) == 2
        assert len(bboxes) == 2
        assert len(points) == 2
        assert len(categories) == 2
        assert len(category_ids) == 2
        assert len(is_reference) == 2
        assert len(n_shot) == 2
        assert len(image_paths) == 2
        assert len(mask_paths) == 2

    def test_batch_single_sample(self, sample_single_instance: Sample) -> None:
        """Test batch with single sample."""
        batch = Batch.collate([sample_single_instance])

        assert len(batch) == 1
        assert batch[0] == sample_single_instance

        # Test all properties work with single sample
        images = batch.images
        assert len(images) == 1
        assert isinstance(images[0], torch.Tensor)

        categories = batch.categories
        assert len(categories) == 1
        assert categories[0] == ["cat"]

    def test_batch_large_batch(self) -> None:
        """Test batch with many samples."""
        samples = []
        for i in range(10):
            sample = Sample(
                image=_rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                image_path=f"image_{i}.jpg",
                masks=_rng.integers(0, 2, (1, 224, 224), dtype=np.uint8),
                categories=[f"category_{i}"],
                category_ids=np.array([i], dtype=np.int32),
                is_reference=[i % 2 == 0],
                n_shot=[i % 3],
            )
            samples.append(sample)

        batch = Batch.collate(samples)

        assert len(batch) == 10

        # Test all properties work with large batch
        images = batch.images
        assert len(images) == 10

        categories = batch.categories
        assert len(categories) == 10
        assert categories[0] == ["category_0"]
        assert categories[9] == ["category_9"]
