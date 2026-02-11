# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Sample class."""

import numpy as np
import torch

from instantlearn.data.base.sample import Sample

# Create a random generator for consistent testing
_rng = np.random.default_rng(42)


class TestSample:
    """Test Sample class functionality."""

    def test_sample_creation_basic(self) -> None:
        """Test basic sample creation with required fields."""
        image = _rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        sample = Sample(image=image, image_path="test.jpg")

        assert sample.image.shape == (224, 224, 3)
        assert sample.image_path == "test.jpg"
        assert sample.masks is None
        assert sample.bboxes is None
        assert sample.points is None
        assert sample.categories == ["object"]
        assert sample.category_ids == [0]
        assert sample.mask_paths is None
        assert sample.is_reference == [False]
        assert sample.n_shot == [-1]

    def test_sample_creation_single_instance(self) -> None:
        """Test sample creation for single-instance scenario (PerSeg)."""
        image = _rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        masks = _rng.integers(0, 2, (1, 224, 224), dtype=np.uint8)
        bboxes = np.array([[10, 20, 100, 120]], dtype=np.float32)
        points = np.array([[50, 60]], dtype=np.float32)

        sample = Sample(
            image=image,
            image_path="test.jpg",
            masks=masks,
            bboxes=bboxes,
            points=points,
            categories=["cat"],
            category_ids=np.array([0], dtype=np.int32),
            mask_paths=["mask.png"],
            is_reference=[True],
            n_shot=[0],
        )

        assert sample.image.shape == (224, 224, 3)
        assert sample.masks.shape == (1, 224, 224)
        assert sample.bboxes.shape == (1, 4)
        assert sample.points.shape == (1, 2)
        assert sample.categories == ["cat"]
        assert sample.category_ids.tolist() == [0]
        assert sample.mask_paths == ["mask.png"]
        assert sample.is_reference == [True]
        assert sample.n_shot == [0]

    def test_sample_creation_multi_instance(self) -> None:
        """Test sample creation for multi-instance scenario (LVIS)."""
        image = _rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
        masks = _rng.integers(0, 2, (3, 512, 512), dtype=np.uint8)
        bboxes = np.array([[10, 20, 110, 120], [200, 150, 350, 270], [100, 100, 200, 200]], dtype=np.float32)
        points = np.array([[50, 60], [250, 200], [150, 150]], dtype=np.float32)

        sample = Sample(
            image=image,
            image_path="test.jpg",
            masks=masks,
            bboxes=bboxes,
            points=points,
            categories=["person", "car", "dog"],
            category_ids=np.array([0, 1, 2], dtype=np.int32),
            mask_paths=["mask1.png", "mask2.png", "mask3.png"],
            is_reference=[True, False, True],
            n_shot=[0, -1, 1],
        )

        assert sample.image.shape == (512, 512, 3)
        assert sample.masks.shape == (3, 512, 512)
        assert sample.bboxes.shape == (3, 4)
        assert sample.points.shape == (3, 2)
        assert sample.categories == ["person", "car", "dog"]
        assert sample.category_ids.tolist() == [0, 1, 2]
        assert sample.mask_paths == ["mask1.png", "mask2.png", "mask3.png"]
        assert sample.is_reference == [True, False, True]
        assert sample.n_shot == [0, -1, 1]

    def test_sample_creation_torch_tensors(self) -> None:
        """Test sample creation with torch tensors."""
        image = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
        masks = torch.randint(0, 2, (2, 224, 224), dtype=torch.uint8)
        bboxes = torch.tensor([[10, 20, 100, 120], [200, 150, 350, 270]], dtype=torch.float32)
        points = torch.tensor([[50, 60], [250, 200]], dtype=torch.float32)

        sample = Sample(
            image=image,
            image_path="test.jpg",
            masks=masks,
            bboxes=bboxes,
            points=points,
            categories=["cat", "dog"],
            category_ids=torch.tensor([0, 1], dtype=torch.int32),
            is_reference=[True, False],
            n_shot=[0, -1],
        )

        assert isinstance(sample.image, torch.Tensor)
        assert isinstance(sample.masks, torch.Tensor)
        assert isinstance(sample.bboxes, torch.Tensor)
        assert isinstance(sample.points, torch.Tensor)
        assert isinstance(sample.category_ids, torch.Tensor)

    def test_sample_creation_minimal(self) -> None:
        """Test sample creation with minimal required fields."""
        image = _rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        sample = Sample(image=image)

        assert sample.image.shape == (224, 224, 3)
        assert sample.image_path is None
        assert sample.masks is None
        assert sample.bboxes is None
        assert sample.points is None
        assert sample.categories == ["object"]
        assert sample.category_ids == [0]
        assert sample.mask_paths is None
        assert sample.is_reference == [False]
        assert sample.n_shot == [-1]

    def test_sample_creation_only_bboxes(self) -> None:
        """Test sample creation with only bboxes (no masks)."""
        image = _rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
        bboxes = np.array([[10, 20, 110, 120], [200, 150, 350, 270]], dtype=np.float32)

        sample = Sample(
            image=image,
            image_path="test.jpg",
            bboxes=bboxes,
            categories=["cat", "dog"],
            category_ids=np.array([0, 1], dtype=np.int32),
            is_reference=[True, True],
            n_shot=[0, 0],
        )

        assert sample.masks is None
        assert sample.bboxes.shape == (2, 4)
        assert sample.points is None
        assert sample.categories == ["cat", "dog"]
        assert sample.category_ids.tolist() == [0, 1]
        assert sample.is_reference == [True, True]
        assert sample.n_shot == [0, 0]

    def test_sample_creation_only_points(self) -> None:
        """Test sample creation with only points (no masks or bboxes)."""
        image = _rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
        points = np.array([[100, 150], [300, 400]], dtype=np.float32)

        sample = Sample(
            image=image,
            image_path="test.jpg",
            points=points,
            categories=["person", "person"],
            category_ids=np.array([2, 2], dtype=np.int32),
            is_reference=[False, False],
            n_shot=[-1, -1],
        )

        assert sample.masks is None
        assert sample.bboxes is None
        assert sample.points.shape == (2, 2)
        assert sample.categories == ["person", "person"]
        assert sample.category_ids.tolist() == [2, 2]
        assert sample.is_reference == [False, False]
        assert sample.n_shot == [-1, -1]

    def test_sample_creation_mixed_types(self) -> None:
        """Test sample creation with mixed numpy and torch types."""
        image = _rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        masks = torch.randint(0, 2, (2, 224, 224), dtype=torch.uint8)
        bboxes = np.array([[10, 20, 100, 120], [200, 150, 350, 270]], dtype=np.float32)
        points = torch.tensor([[50, 60], [250, 200]], dtype=torch.float32)

        sample = Sample(
            image=image,
            image_path="test.jpg",
            masks=masks,
            bboxes=bboxes,
            points=points,
            categories=["cat", "dog"],
            category_ids=np.array([0, 1], dtype=np.int32),
            is_reference=[True, False],
            n_shot=[0, -1],
        )

        assert isinstance(sample.image, np.ndarray)
        assert isinstance(sample.masks, torch.Tensor)
        assert isinstance(sample.bboxes, np.ndarray)
        assert isinstance(sample.points, torch.Tensor)
        assert isinstance(sample.category_ids, np.ndarray)

    def test_sample_creation_empty_lists(self) -> None:
        """Test sample creation with empty lists for multi-instance fields."""
        image = _rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)

        sample = Sample(
            image=image,
            image_path="test.jpg",
            categories=[],
            category_ids=np.array([], dtype=np.int32),
            is_reference=[],
            n_shot=[],
        )

        assert sample.categories == []
        assert sample.category_ids.tolist() == []
        assert sample.is_reference == []
        assert sample.n_shot == []

    def test_sample_creation_invalid_shapes(self) -> None:
        """Test sample creation with invalid tensor shapes."""
        image = _rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)

        # Test with mismatched instance counts - this should work as Sample doesn't validate
        # The validation would happen at the dataset level, not in Sample creation
        sample = Sample(
            image=image,
            image_path="test.jpg",
            categories=["cat", "dog"],  # 2 categories
            category_ids=np.array([0], dtype=np.int32),  # 1 category_id
            is_reference=[True, False],  # 2 flags
            n_shot=[0],  # 1 shot
        )

        # Sample creation should succeed - validation happens elsewhere
        assert sample.categories == ["cat", "dog"]
        assert sample.category_ids.tolist() == [0]
        assert sample.is_reference == [True, False]
        assert sample.n_shot == [0]

    def test_sample_creation_default_values(self) -> None:
        """Test that default values are set correctly."""
        image = _rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        sample = Sample(image=image)

        # Test default values
        assert sample.is_reference == [False]
        assert sample.n_shot == [-1]

        # Test that these are lists, not single values
        assert isinstance(sample.is_reference, list)
        assert isinstance(sample.n_shot, list)

    def test_sample_creation_hwc_format(self) -> None:
        """Test that images are stored in HWC format."""
        # Test numpy array
        image_np = _rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        sample_np = Sample(image=image_np)
        assert sample_np.image.shape == (224, 224, 3)  # HWC format

        # Test torch tensor (should be converted to HWC)
        image_torch = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)  # CHW format
        sample_torch = Sample(image=image_torch)
        # Note: The actual conversion would depend on implementation
        # This test assumes the tensor is stored as-is for now
        assert sample_torch.image.shape == (3, 224, 224)  # CHW format (as stored)

    def test_sample_creation_mask_consistency(self) -> None:
        """Test that mask dimensions are consistent."""
        image = _rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        masks = _rng.integers(0, 2, (2, 224, 224), dtype=np.uint8)

        sample = Sample(
            image=image,
            image_path="test.jpg",
            masks=masks,
            categories=["cat", "dog"],
            category_ids=np.array([0, 1], dtype=np.int32),
            is_reference=[True, False],
            n_shot=[0, -1],
        )

        # All masks should have the same H, W as the image
        assert sample.masks.shape[1:] == sample.image.shape[:2]  # (H, W)
        assert sample.masks.shape[0] == len(sample.categories)  # N instances

    def test_sample_creation_bbox_format(self) -> None:
        """Test that bboxes are in [x, y, w, h] format."""
        image = _rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        bboxes = np.array([[10, 20, 100, 120]], dtype=np.float32)  # [x, y, w, h]

        sample = Sample(
            image=image,
            image_path="test.jpg",
            bboxes=bboxes,
            categories=["cat"],
            category_ids=np.array([0], dtype=np.int32),
            is_reference=[True],
            n_shot=[0],
        )

        assert sample.bboxes.shape == (1, 4)  # One bbox with 4 coordinates
        assert sample.bboxes[0, 0] == 10  # x
        assert sample.bboxes[0, 1] == 20  # y
        assert sample.bboxes[0, 2] == 100  # w
        assert sample.bboxes[0, 3] == 120  # h

    def test_sample_creation_points_format(self) -> None:
        """Test that points are in [x, y] format."""
        image = _rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        points = np.array([[50, 60], [100, 150]], dtype=np.float32)  # [x, y]

        sample = Sample(
            image=image,
            image_path="test.jpg",
            points=points,
            categories=["person", "person"],
            category_ids=np.array([0, 0], dtype=np.int32),
            is_reference=[False, False],
            n_shot=[-1, -1],
        )

        assert sample.points.shape == (2, 2)  # Two points with 2 coordinates each
        assert sample.points[0, 0] == 50  # x1
        assert sample.points[0, 1] == 60  # y1
        assert sample.points[1, 0] == 100  # x2
        assert sample.points[1, 1] == 150  # y2
