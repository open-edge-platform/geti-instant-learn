# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MultiInstancePriorFilter."""

import pytest
import torch

from instantlearn.models.grounded_sam import BoxPromptFilter


def test_multi_instance_prior_filter_large_boxes_filtered() -> None:
    """Test that large boxes containing smaller boxes are filtered out."""
    filt = BoxPromptFilter()

    # Scenario: one large box contains three smaller boxes that almost fill it.
    # The large box should be filtered out.
    # box_prompts: [T=1, C=1, max_boxes=4, 5]
    box_prompts = torch.zeros(1, 1, 4, 5, dtype=torch.float32)

    # Large box: Area = 10000
    box_prompts[0, 0, 0] = torch.tensor([10, 10, 110, 110, 0.9])
    # Small boxes (contained within large box)
    box_prompts[0, 0, 1] = torch.tensor([11, 11, 50, 50, 0.8])  # Area: 1521
    box_prompts[0, 0, 2] = torch.tensor([51, 11, 109, 50, 0.8])  # Area: 2262
    box_prompts[0, 0, 3] = torch.tensor([11, 51, 109, 109, 0.8])  # Area: 5684
    # Total contained area: 9467 > 0.8 * 10000 = 8000, so large box should be filtered

    filtered_prompts = filt(box_prompts)

    # Only the 3 small boxes should remain (check by counting non-zero scores)
    remaining_boxes = filtered_prompts[0, 0]
    num_remaining = (remaining_boxes[:, -1] != 0).sum().item()
    expected_count = 3
    pytest.assume(num_remaining == expected_count)

    # Verify the remaining boxes are the small ones
    # Check that the large box (10, 10, 110, 110) is not in the remaining boxes
    large_box_coords = torch.tensor([10, 10, 110, 110]).float()
    for i in range(expected_count):
        pytest.assume(not torch.allclose(remaining_boxes[i, :4], large_box_coords))


def test_multi_instance_prior_filter_small_contained_boxes_kept() -> None:
    """Test that boxes with small contained boxes are not filtered."""
    filt = BoxPromptFilter()

    # Scenario: large box with small contained box that doesn't exceed threshold
    # box_prompts: [T=1, C=1, max_boxes=2, 5]
    box_prompts = torch.zeros(1, 1, 2, 5, dtype=torch.float32)

    # Large box: Area = 10000
    box_prompts[0, 0, 0] = torch.tensor([0, 0, 100, 100, 0.9])
    # Small box: Area = 100 (only 1% of large box, well below 80% threshold)
    box_prompts[0, 0, 1] = torch.tensor([10, 10, 20, 20, 0.8])

    filtered_prompts = filt(box_prompts)

    # Both boxes should be kept (check by counting non-zero scores)
    num_remaining = (filtered_prompts[0, 0, :, -1] != 0).sum().item()
    expected_count = 2
    pytest.assume(num_remaining == expected_count)


def test_multi_instance_prior_filter_multiple_images() -> None:
    """Test filtering across multiple images."""
    filt = BoxPromptFilter()

    # 2 images, 1 category, max 4 boxes
    box_prompts = torch.zeros(2, 1, 4, 5, dtype=torch.float32)

    # Image 1: Large box with small contained boxes (should be filtered)
    box_prompts[0, 0, 0] = torch.tensor([10, 10, 110, 110, 0.9])  # Large box
    box_prompts[0, 0, 1] = torch.tensor([11, 11, 50, 50, 0.8])
    box_prompts[0, 0, 2] = torch.tensor([51, 11, 109, 50, 0.8])
    box_prompts[0, 0, 3] = torch.tensor([11, 51, 109, 109, 0.8])

    # Image 2: Large box with small contained box (should not be filtered)
    box_prompts[1, 0, 0] = torch.tensor([0, 0, 100, 100, 0.9])
    box_prompts[1, 0, 1] = torch.tensor([10, 10, 20, 20, 0.8])

    filtered_prompts = filt(box_prompts)

    # Image 1: 3 small boxes remain (large box filtered)
    num_remaining_img1 = (filtered_prompts[0, 0, :, -1] != 0).sum().item()
    pytest.assume(num_remaining_img1 == 3)
    # Image 2: Both boxes remain
    num_remaining_img2 = (filtered_prompts[1, 0, :, -1] != 0).sum().item()
    pytest.assume(num_remaining_img2 == 2)


def test_multi_instance_prior_filter_empty_input() -> None:
    """Test filtering with zero boxes."""
    filt = BoxPromptFilter()

    box_prompts = torch.zeros(1, 1, 4, 5, dtype=torch.float32)

    filtered_prompts = filt(box_prompts)

    # All boxes should still be zero (no valid boxes)
    num_remaining = (filtered_prompts[0, 0, :, -1] != 0).sum().item()
    pytest.assume(num_remaining == 0)


def test_multi_instance_prior_filter_multiple_categories() -> None:
    """Test filtering with multiple categories."""
    filt = BoxPromptFilter()

    # 1 image, 2 categories, max 4 boxes
    box_prompts = torch.zeros(1, 2, 4, 5, dtype=torch.float32)

    # Category 0: Large box with contained boxes (should be filtered)
    box_prompts[0, 0, 0] = torch.tensor([10, 10, 110, 110, 0.9])
    box_prompts[0, 0, 1] = torch.tensor([11, 11, 50, 50, 0.8])
    box_prompts[0, 0, 2] = torch.tensor([51, 11, 109, 50, 0.8])
    box_prompts[0, 0, 3] = torch.tensor([11, 51, 109, 109, 0.8])

    # Category 1: Single box (should remain)
    box_prompts[0, 1, 0] = torch.tensor([0, 0, 50, 50, 0.9])

    filtered_prompts = filt(box_prompts)

    # Category 0: 3 small boxes remain
    num_remaining_cat0 = (filtered_prompts[0, 0, :, -1] != 0).sum().item()
    pytest.assume(num_remaining_cat0 == 3)
    # Category 1: 1 box remains
    num_remaining_cat1 = (filtered_prompts[0, 1, :, -1] != 0).sum().item()
    pytest.assume(num_remaining_cat1 == 1)
