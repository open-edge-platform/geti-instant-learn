# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MultiInstancePriorFilter."""

import pytest
import torch

from getiprompt.components.filters import BoxPromptFilter


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

    num_boxes = torch.tensor([[4]], dtype=torch.int64)

    filtered_prompts, filtered_num = filt(box_prompts, num_boxes)

    # Only the 3 small boxes should remain
    expected_count = 3
    pytest.assume(filtered_num[0, 0].item() == expected_count)

    # Verify the remaining boxes are the small ones
    remaining_boxes = filtered_prompts[0, 0, :expected_count]
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

    num_boxes = torch.tensor([[2]], dtype=torch.int64)

    filtered_prompts, filtered_num = filt(box_prompts, num_boxes)

    # Both boxes should be kept
    expected_count = 2
    pytest.assume(filtered_num[0, 0].item() == expected_count)


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

    num_boxes = torch.tensor([[4], [2]], dtype=torch.int64)

    filtered_prompts, filtered_num = filt(box_prompts, num_boxes)

    # Image 1: 3 small boxes remain (large box filtered)
    pytest.assume(filtered_num[0, 0].item() == 3)
    # Image 2: Both boxes remain
    pytest.assume(filtered_num[1, 0].item() == 2)


def test_multi_instance_prior_filter_empty_input() -> None:
    """Test filtering with zero boxes."""
    filt = BoxPromptFilter()

    box_prompts = torch.zeros(1, 1, 4, 5, dtype=torch.float32)
    num_boxes = torch.tensor([[0]], dtype=torch.int64)

    filtered_prompts, filtered_num = filt(box_prompts, num_boxes)

    pytest.assume(filtered_num[0, 0].item() == 0)


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

    num_boxes = torch.tensor([[4, 1]], dtype=torch.int64)

    filtered_prompts, filtered_num = filt(box_prompts, num_boxes)

    # Category 0: 3 small boxes remain
    pytest.assume(filtered_num[0, 0].item() == 3)
    # Category 1: 1 box remains
    pytest.assume(filtered_num[0, 1].item() == 1)
