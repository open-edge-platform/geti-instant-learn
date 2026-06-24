# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Batch class."""

import numpy as np
import pytest
import torch

from instantlearn.data.base.batch import Batch
from instantlearn.data.base.sample import Category, Sample

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
        categories=[Category(0, "cat")],
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
        categories=[Category(0, "person"), Category(1, "car"), Category(2, "dog")],
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
        categories=[Category(0, "cat")],
        is_reference=[True],
        n_shot=[0],
    )


class TestBatchBasic:  # noqa: PLR0904 Too many public methods (21 > 20)
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

    def test_batch_collate_single_sample(self, sample_single_instance: Sample) -> None:
        """Test that collating a single sample (not wrapped in list) works."""
        batch = Batch.collate(sample_single_instance)

        assert len(batch) == 1
        assert batch[0] == sample_single_instance
        assert batch.samples == [sample_single_instance]

    def test_batch_collate_single_sample_equivalence(self, sample_single_instance: Sample) -> None:
        """Test that collating single sample is equivalent to wrapping in list."""
        batch1 = Batch.collate(sample_single_instance)
        batch2 = Batch.collate([sample_single_instance])

        assert len(batch1) == len(batch2) == 1
        assert batch1[0] == batch2[0] == sample_single_instance
        assert batch1.samples == batch2.samples == [sample_single_instance]

    def test_batch_collate_single_sample_properties(self, sample_single_instance: Sample) -> None:
        """Test that all properties work when collating a single sample."""
        batch = Batch.collate(sample_single_instance)

        # Test all properties work with single sample
        images = batch.images
        assert len(images) == 1
        assert isinstance(images[0], np.ndarray)

        masks = batch.masks
        assert len(masks) == 1
        assert isinstance(masks[0], np.ndarray)

        bboxes = batch.bboxes
        assert len(bboxes) == 1
        assert isinstance(bboxes[0], np.ndarray)

        points = batch.points
        assert len(points) == 1
        assert isinstance(points[0], np.ndarray)

        category_labels = batch.category_labels
        assert len(category_labels) == 1
        assert category_labels[0] == ["cat"]

        label_ids = batch.label_ids
        assert len(label_ids) == 1
        assert label_ids[0] == [0]

        is_reference = batch.is_reference
        assert len(is_reference) == 1
        assert is_reference[0] == [True]

        n_shot = batch.n_shot
        assert len(n_shot) == 1
        assert n_shot[0] == [0]

        image_paths = batch.image_paths
        assert len(image_paths) == 1
        assert image_paths[0] == "single.jpg"

        mask_paths = batch.mask_paths
        assert len(mask_paths) == 1
        assert mask_paths[0] == ["mask1.png"]

    def test_batch_collate_single_multi_instance_sample(self, sample_multi_instance: Sample) -> None:
        """Test that collating a single multi-instance sample works."""
        batch = Batch.collate(sample_multi_instance)

        assert len(batch) == 1
        assert batch[0] == sample_multi_instance
        assert batch.samples == [sample_multi_instance]

        # Test multi-instance properties
        category_labels = batch.category_labels
        assert len(category_labels) == 1
        assert category_labels[0] == ["person", "car", "dog"]

        label_ids = batch.label_ids
        assert len(label_ids) == 1
        assert label_ids[0] == [0, 1, 2]

        is_reference = batch.is_reference
        assert len(is_reference) == 1
        assert is_reference[0] == [True, False, True]

        n_shot = batch.n_shot
        assert len(n_shot) == 1
        assert n_shot[0] == [0, -1, 1]

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
        assert all(isinstance(img, np.ndarray) for img in images)

        # Backend-neutral: arrays are returned without copying
        images_again = batch.images
        assert images_again[0] is images[0]  # Same underlying array

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
        assert isinstance(masks[0], np.ndarray)  # Has masks
        assert isinstance(masks[1], np.ndarray)  # Has masks
        assert masks[2] is None  # No masks

        # Backend-neutral: arrays are returned without copying
        masks_again = batch.masks
        assert masks_again[0] is masks[0]  # Same underlying array

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
        assert isinstance(bboxes[0], np.ndarray)  # Has bboxes
        assert isinstance(bboxes[1], np.ndarray)  # Has bboxes
        assert bboxes[2] is None  # No bboxes (sample_no_masks doesn't have bboxes)

    def test_batch_points_property(self, sample_single_instance: Sample, sample_multi_instance: Sample) -> None:
        """Test batch points property."""
        samples = [sample_single_instance, sample_multi_instance]
        batch = Batch.collate(samples)

        points = batch.points
        assert len(points) == 2
        assert all(isinstance(pts, np.ndarray) for pts in points)

    def test_batch_categories_property(self, sample_single_instance: Sample, sample_multi_instance: Sample) -> None:
        """Test batch categories property."""
        samples = [sample_single_instance, sample_multi_instance]
        batch = Batch.collate(samples)

        category_labels = batch.category_labels
        assert len(category_labels) == 2
        assert category_labels[0] == ["cat"]  # Single instance
        assert category_labels[1] == ["person", "car", "dog"]  # Multi instance

    def test_batch_category_ids_property(self, sample_single_instance: Sample, sample_multi_instance: Sample) -> None:
        """Test batch label_ids property."""
        samples = [sample_single_instance, sample_multi_instance]
        batch = Batch.collate(samples)

        label_ids = batch.label_ids
        assert len(label_ids) == 2
        assert all(isinstance(ids, list) for ids in label_ids)
        assert label_ids[0] == [0]  # Single instance
        assert label_ids[1] == [0, 1, 2]  # Multi instance

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
        """Test that numpy arrays are returned as-is (Batch is backend-neutral)."""
        batch = Batch.collate([sample_single_instance])

        # Image returned as stored numpy array
        images = batch.images
        assert isinstance(images[0], np.ndarray)
        assert images[0].shape == (224, 224, 3)  # HWC format (as stored in Sample)

        # Masks returned as stored numpy array
        masks = batch.masks
        assert isinstance(masks[0], np.ndarray)
        assert masks[0].shape == (1, 224, 224)

        # Bboxes returned as stored numpy array
        bboxes = batch.bboxes
        assert isinstance(bboxes[0], np.ndarray)
        assert bboxes[0].shape == (1, 4)

        # Points returned as stored numpy array
        points = batch.points
        assert isinstance(points[0], np.ndarray)
        assert points[0].shape == (1, 2)

        # Test label_ids (backend-neutral: plain list of ints)
        label_ids = batch.label_ids
        assert label_ids[0] == [0]


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
            categories=[Category(0, "cat")],
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

        label_ids = batch.label_ids
        assert label_ids[0] == sample.label_ids  # Same values

    def test_batch_mixed_numpy_torch(self) -> None:
        """Test batch with mixed numpy and torch tensors."""
        # Create samples with different tensor types
        sample_numpy = Sample(
            image=_rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
            image_path="numpy.jpg",
            masks=_rng.integers(0, 2, (1, 224, 224), dtype=np.uint8),
            categories=[Category(0, "cat")],
            is_reference=[True],
            n_shot=[0],
        )

        sample_torch = Sample(
            image=torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8),
            image_path="torch.jpg",
            masks=torch.randint(0, 2, (1, 224, 224), dtype=torch.uint8),
            categories=[Category(1, "dog")],
            is_reference=[False],
            n_shot=[-1],
        )

        batch = Batch.collate([sample_numpy, sample_torch])

        # Backend-neutral: arrays returned as stored (numpy stays numpy, torch stays torch)
        images = batch.images
        assert isinstance(images[0], np.ndarray)  # numpy sample
        assert isinstance(images[1], torch.Tensor)  # torch sample

        masks = batch.masks
        assert isinstance(masks[0], np.ndarray)  # numpy sample
        assert isinstance(masks[1], torch.Tensor)  # torch sample

        label_ids = batch.label_ids
        assert label_ids[0] == [0]  # numpy sample
        assert label_ids[1] == [1]  # torch sample

    def test_batch_empty_tensors(self) -> None:
        """Test batch with empty tensors."""
        sample = Sample(
            image=_rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
            image_path="empty.jpg",
            categories=[],
            is_reference=[],
            n_shot=[],
        )

        batch = Batch.collate([sample])

        # Test empty label_ids
        label_ids = batch.label_ids
        assert label_ids[0] == []

        # Test empty categories
        category_labels = batch.category_labels
        assert category_labels[0] == []

    def test_batch_lazy_conversion(self, sample_single_instance: Sample) -> None:
        """Test that tensor conversion is lazy."""
        batch = Batch.collate([sample_single_instance])

        # Access images property multiple times
        images1 = batch.images
        images2 = batch.images

        # Backend-neutral: underlying arrays are not copied between accesses
        assert images1[0] is images2[0]

        # Access masks property multiple times
        masks1 = batch.masks
        masks2 = batch.masks

        # Backend-neutral: underlying arrays are not copied between accesses
        assert masks1[0] is masks2[0]

    def test_batch_property_independence(self, sample_single_instance: Sample, sample_multi_instance: Sample) -> None:
        """Test that different properties are independent."""
        batch = Batch.collate([sample_single_instance, sample_multi_instance])

        # Access different properties
        images = batch.images
        masks = batch.masks
        bboxes = batch.bboxes
        points = batch.points
        categories = batch.categories
        label_ids = batch.label_ids
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
        assert len(label_ids) == 2
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
        assert isinstance(images[0], np.ndarray)

        category_labels = batch.category_labels
        assert len(category_labels) == 1
        assert category_labels[0] == ["cat"]

    def test_batch_large_batch(self) -> None:
        """Test batch with many samples."""
        samples = []
        for i in range(10):
            sample = Sample(
                image=_rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                image_path=f"image_{i}.jpg",
                masks=_rng.integers(0, 2, (1, 224, 224), dtype=np.uint8),
                categories=[Category(i, f"category_{i}")],
                is_reference=[i % 2 == 0],
                n_shot=[i % 3],
            )
            samples.append(sample)

        batch = Batch.collate(samples)

        assert len(batch) == 10

        # Test all properties work with large batch
        images = batch.images
        assert len(images) == 10

        category_labels = batch.category_labels
        assert len(category_labels) == 10
        assert category_labels[0] == ["category_0"]
        assert category_labels[9] == ["category_9"]
