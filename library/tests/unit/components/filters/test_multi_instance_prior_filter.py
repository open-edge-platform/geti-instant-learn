# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MultiInstancePriorFilter."""

import pytest
import torch

from getiprompt.components.filters import BoxPromptFilter


def test_multi_instance_prior_filter() -> None:
    """Test the MultiInstancePriorFilter."""
    filt = BoxPromptFilter()

    # Create a scenario where one large box contains three smaller boxes
    # that almost fill it. The large box should be filtered out.
    boxes1 = {0: []}
    # Large box
    large_box1 = torch.tensor([[10, 10, 110, 110, 0.9, 1]])  # Area: 10000
    # Small boxes
    small_boxes1 = torch.tensor([
        [11, 11, 50, 50, 0.8, 1],  # Area: 1521
        [51, 11, 109, 50, 0.8, 1],  # Area: 2262
        [11, 51, 109, 109, 0.8, 1],  # Area: 5684
    ])
    # Total area: 9467. This should trigger the filter for the large box.
    boxes1[0].append(torch.cat([large_box1, small_boxes1]))
    # Convert Boxes to dict format expected by BoxPromptFilter
    prompts1: dict[int, torch.Tensor] = {0: boxes1[0][0]}

    # Create a second scenario where the contained boxes are too small
    # to trigger the filter.
    boxes2 = {0: []}
    large_box2 = torch.tensor([[0, 0, 100, 100, 0.9, 1]])
    small_boxes2 = torch.tensor([[10, 10, 20, 20, 0.8, 1]])
    boxes2[0].append(torch.cat([large_box2, small_boxes2]))
    # Convert Boxes to dict format expected by BoxPromptFilter
    prompts2: dict[int, torch.Tensor] = {0: boxes2[0][0]}

    filtered_prompts_list = filt([prompts1, prompts2])

    # In the first case, only the 3 small boxes should remain.
    remaining_boxes1 = filtered_prompts_list[0][0]
    expected_value = 3
    pytest.assume(remaining_boxes1.shape[0] == expected_value)
    # Sort boxes before comparison to avoid order-related failures
    remaining_boxes1_sorted, _ = torch.sort(remaining_boxes1, dim=0)
    small_boxes1_sorted, _ = torch.sort(small_boxes1, dim=0)
    pytest.assume(torch.all(remaining_boxes1_sorted == small_boxes1_sorted))

    # In the second case, both boxes should be kept.
    remaining_boxes2 = filtered_prompts_list[1][0]
    expected_boxes2 = torch.cat([large_box2, small_boxes2])
    expected_value = 2
    pytest.assume(remaining_boxes2.shape[0] == expected_value)
    # Sort boxes before comparison
    remaining_boxes2_sorted, _ = torch.sort(remaining_boxes2, dim=0)
    expected_boxes2_sorted, _ = torch.sort(expected_boxes2, dim=0)
    pytest.assume(torch.all(remaining_boxes2_sorted == expected_boxes2_sorted))
