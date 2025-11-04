# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ClassOverlapMaskFilter."""

import pytest
import torch

from getiprompt.components.filters.mask_filter import ClassOverlapMaskFilter
from getiprompt.types import Masks, Points


class TestClassOverlapMaskFilterInitialization:
    """Test ClassOverlapMaskFilter initialization."""

    def test_initialization_default(self) -> None:
        """Test initialization with default parameters."""
        filter_module = ClassOverlapMaskFilter()
        assert filter_module is not None
        assert isinstance(filter_module, ClassOverlapMaskFilter)

    def test_initialization_custom_threshold(self) -> None:
        """Test initialization with custom threshold."""
        filter_module = ClassOverlapMaskFilter(threshold_iou=0.5)
        assert filter_module.threshold_iou == 0.5


class TestClassOverlapMaskFilterForward:
    """Test forward pass functionality of ClassOverlapMaskFilter."""

    @pytest.fixture
    def sample_masks(self) -> list[Masks]:
        """Create sample masks for testing."""
        masks_list = []

        # Create masks for first image
        masks1 = Masks()
        mask1 = torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        mask2 = torch.tensor(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        masks1.add(mask1, class_id=0)
        masks1.add(mask2, class_id=1)

        # Create masks for second image
        masks2 = Masks()
        mask3 = torch.tensor(
            [
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        masks2.add(mask3, class_id=0)

        masks_list.extend((masks1, masks2))

        return masks_list

    @pytest.fixture
    def sample_points(self) -> list[Points]:
        """Create sample points for testing."""
        points_list = []

        # Create points for first image
        points1 = Points()
        point1 = torch.tensor([[1, 1, 0.9, 0]], dtype=torch.float32)  # [x, y, score, label]
        point2 = torch.tensor([[2, 2, 0.8, 1]], dtype=torch.float32)
        points1.add(point1, class_id=0)
        points1.add(point2, class_id=1)

        # Create points for second image
        points2 = Points()
        point3 = torch.tensor([[1, 1, 0.7, 0]], dtype=torch.float32)
        points2.add(point3, class_id=0)

        points_list.extend((points1, points2))

        return points_list

    def test_forward_basic(self, sample_masks: list[Masks], sample_points: list[Points]) -> None:
        """Test basic forward pass."""
        filter_module = ClassOverlapMaskFilter(threshold_iou=0.5)

        result_masks, result_points = filter_module.forward(
            all_pred_masks=sample_masks,
            all_pred_points=sample_points,
            threshold_iou=0.5,
        )

        assert isinstance(result_masks, list)
        assert isinstance(result_points, list)
        assert len(result_masks) == len(sample_masks)
        assert len(result_points) == len(sample_points)

        # Check that results are Masks and Points objects
        for mask, point in zip(result_masks, result_points, strict=False):
            assert isinstance(mask, Masks)
            assert isinstance(point, Points)

    def test_forward_with_empty_inputs(self) -> None:
        """Test forward pass with empty inputs."""
        filter_module = ClassOverlapMaskFilter()

        result_masks, result_points = filter_module.forward(
            all_pred_masks=[],
            all_pred_points=[],
            threshold_iou=0.5,
        )

        assert result_masks == []
        assert result_points == []

    def test_forward_with_none_inputs(self) -> None:
        """Test forward pass with None inputs."""
        filter_module = ClassOverlapMaskFilter()

        result_masks, result_points = filter_module.forward(
            all_pred_masks=None,
            all_pred_points=None,
            threshold_iou=0.5,
        )

        assert result_masks == []
        assert result_points == []

    def test_forward_with_empty_masks(self) -> None:
        """Test forward pass with empty masks."""
        filter_module = ClassOverlapMaskFilter()

        # Create empty masks
        empty_masks = Masks()
        empty_points = Points()

        result_masks, result_points = filter_module.forward(
            all_pred_masks=[empty_masks],
            all_pred_points=[empty_points],
            threshold_iou=0.5,
        )

        assert len(result_masks) == 1
        assert len(result_points) == 1
        assert isinstance(result_masks[0], Masks)
        assert isinstance(result_points[0], Points)

    def test_forward_with_single_mask_per_class(self, sample_masks: list[Masks], sample_points: list[Points]) -> None:
        """Test forward pass with single mask per class (no overlap)."""
        filter_module = ClassOverlapMaskFilter(threshold_iou=0.5)

        result_masks, result_points = filter_module.forward(
            all_pred_masks=sample_masks,
            all_pred_points=sample_points,
            threshold_iou=0.5,
        )

        # Should return the same number of masks and points
        assert len(result_masks) == len(sample_masks)
        assert len(result_points) == len(sample_points)

    def test_forward_with_overlapping_masks(self) -> None:
        """Test forward pass with overlapping masks that should be filtered by NMS."""
        filter_module = ClassOverlapMaskFilter(threshold_iou=0.5)

        # Create overlapping masks
        masks = Masks()
        mask1 = torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        mask2 = torch.tensor(
            [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        masks.add(mask1, class_id=0)
        masks.add(mask2, class_id=0)  # Same class, overlapping

        points = Points()
        point1 = torch.tensor([[0, 0, 0.9, 0]], dtype=torch.float32)
        point2 = torch.tensor([[1, 0, 0.8, 0]], dtype=torch.float32)
        points.add(point1, class_id=0)
        points.add(point2, class_id=0)

        result_masks, result_points = filter_module.forward(
            all_pred_masks=[masks],
            all_pred_points=[points],
            threshold_iou=0.5,
        )

        assert len(result_masks) == 1
        assert len(result_points) == 1
        assert isinstance(result_masks[0], Masks)
        assert isinstance(result_points[0], Points)

    def test_forward_with_different_thresholds(self, sample_masks: list[Masks], sample_points: list[Points]) -> None:
        """Test forward pass with different IoU thresholds."""
        filter_module = ClassOverlapMaskFilter()

        # Test with low threshold (more aggressive filtering)
        result_masks_low, result_points_low = filter_module.forward(
            all_pred_masks=sample_masks,
            all_pred_points=sample_points,
            threshold_iou=0.1,
        )

        # Test with high threshold (less aggressive filtering)
        result_masks_high, result_points_high = filter_module.forward(
            all_pred_masks=sample_masks,
            all_pred_points=sample_points,
            threshold_iou=0.9,
        )

        assert len(result_masks_low) == len(result_masks_high)
        assert len(result_points_low) == len(result_points_high)

    def test_forward_preserves_structure(self, sample_masks: list[Masks], sample_points: list[Points]) -> None:
        """Test that forward pass preserves the structure of input data."""
        filter_module = ClassOverlapMaskFilter()

        result_masks, result_points = filter_module.forward(
            all_pred_masks=sample_masks,
            all_pred_points=sample_points,
            threshold_iou=0.5,
        )

        # Check that the number of images is preserved
        assert len(result_masks) == len(sample_masks)
        assert len(result_points) == len(sample_points)

        # Check that each result maintains the same structure
        for i, (result_mask, result_point) in enumerate(zip(result_masks, result_points, strict=False)):
            assert isinstance(result_mask, Masks)
            assert isinstance(result_point, Points)

            # Check that class IDs are preserved
            original_mask_classes = set(sample_masks[i].data.keys())
            result_mask_classes = set(result_mask.data.keys())
            assert result_mask_classes.issubset(original_mask_classes)

    def test_forward_with_mixed_classes(self) -> None:
        """Test forward pass with masks from different classes."""
        filter_module = ClassOverlapMaskFilter(threshold_iou=0.5)

        # Create masks for different classes
        masks = Masks()
        mask1 = torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        mask2 = torch.tensor(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        masks.add(mask1, class_id=0)
        masks.add(mask2, class_id=1)

        points = Points()
        point1 = torch.tensor([[0, 0, 0.9, 0]], dtype=torch.float32)
        point2 = torch.tensor([[2, 0, 0.8, 1]], dtype=torch.float32)
        points.add(point1, class_id=0)
        points.add(point2, class_id=1)

        result_masks, result_points = filter_module.forward(
            all_pred_masks=[masks],
            all_pred_points=[points],
            threshold_iou=0.5,
        )

        assert len(result_masks) == 1
        assert len(result_points) == 1

        # Both classes should be preserved since they don't overlap
        result_mask = result_masks[0]
        assert 0 in result_mask.data
        assert 1 in result_mask.data

    def test_forward_with_scores_and_labels(self) -> None:
        """Test that scores and labels are properly handled in NMS."""
        filter_module = ClassOverlapMaskFilter(threshold_iou=0.5)

        # Create masks with different scores
        masks = Masks()
        mask1 = torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        mask2 = torch.tensor(
            [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        masks.add(mask1, class_id=0)
        masks.add(mask2, class_id=0)

        points = Points()
        # Higher score point
        point1 = torch.tensor([[0, 0, 0.9, 0]], dtype=torch.float32)
        # Lower score point
        point2 = torch.tensor([[1, 0, 0.7, 0]], dtype=torch.float32)
        points.add(point1, class_id=0)
        points.add(point2, class_id=0)

        result_masks, result_points = filter_module.forward(
            all_pred_masks=[masks],
            all_pred_points=[points],
            threshold_iou=0.5,
        )

        assert len(result_masks) == 1
        assert len(result_points) == 1

        # The higher scoring mask should be kept
        result_mask = result_masks[0]
        result_point = result_points[0]
        assert 0 in result_mask.data
        assert 0 in result_point.data

    def test_forward_edge_case_single_mask(self) -> None:
        """Test forward pass with single mask (no NMS needed)."""
        filter_module = ClassOverlapMaskFilter(threshold_iou=0.5)

        masks = Masks()
        mask = torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        masks.add(mask, class_id=0)

        points = Points()
        point = torch.tensor([[0, 0, 0.9, 0]], dtype=torch.float32)
        points.add(point, class_id=0)

        result_masks, result_points = filter_module.forward(
            all_pred_masks=[masks],
            all_pred_points=[points],
            threshold_iou=0.5,
        )

        assert len(result_masks) == 1
        assert len(result_points) == 1
        assert 0 in result_masks[0].data
        assert 0 in result_points[0].data

    def test_forward_with_torchvision_ops(self) -> None:
        """Test that the forward pass uses torchvision ops correctly."""
        filter_module = ClassOverlapMaskFilter(threshold_iou=0.5)

        # Create test data that will exercise the torchvision ops
        masks = Masks()
        mask1 = torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        mask2 = torch.tensor(
            [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        masks.add(mask1, class_id=0)
        masks.add(mask2, class_id=0)

        points = Points()
        point1 = torch.tensor([[0, 0, 0.9, 0]], dtype=torch.float32)
        point2 = torch.tensor([[1, 0, 0.8, 0]], dtype=torch.float32)
        points.add(point1, class_id=0)
        points.add(point2, class_id=0)

        # This should not raise any errors related to torchvision ops
        result_masks, result_points = filter_module.forward(
            all_pred_masks=[masks],
            all_pred_points=[points],
            threshold_iou=0.5,
        )

        assert len(result_masks) == 1
        assert len(result_points) == 1
