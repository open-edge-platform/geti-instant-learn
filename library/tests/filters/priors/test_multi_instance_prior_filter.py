# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from getiprompt.filters.priors import MultiInstancePriorFilter
from getiprompt.types import Boxes, Priors


def test_multi_instance_prior_filter() -> None:
    """Test the MultiInstancePriorFilter."""
    filt = MultiInstancePriorFilter()

    # Create a scenario where one large box contains three smaller boxes
    # that almost fill it. The large box should be filtered out.
    boxes1 = Boxes()
    # Large box
    large_box1 = torch.tensor([[10, 10, 110, 110, 0.9, 1]])  # Area: 10000
    # Small boxes
    small_boxes1 = torch.tensor([
        [11, 11, 50, 50, 0.8, 1],  # Area: 1521
        [51, 11, 109, 50, 0.8, 1],  # Area: 2262
        [11, 51, 109, 109, 0.8, 1],  # Area: 5684
    ])
    # Total area: 9467. This should trigger the filter for the large box.
    boxes1.add(torch.cat([large_box1, small_boxes1]))
    priors1 = Priors(boxes=boxes1)

    # Create a second scenario where the contained boxes are too small
    # to trigger the filter.
    boxes2 = Boxes()
    large_box2 = torch.tensor([[0, 0, 100, 100, 0.9, 1]])
    small_boxes2 = torch.tensor([[10, 10, 20, 20, 0.8, 1]])
    boxes2.add(torch.cat([large_box2, small_boxes2]))
    priors2 = Priors(boxes=boxes2)

    filtered_priors_list = filt([priors1, priors2])

    # In the first case, only the 3 small boxes should remain.
    remaining_boxes1 = filtered_priors_list[0].boxes.get(0)[0]
    assert remaining_boxes1.shape[0] == 3
    # Sort boxes before comparison to avoid order-related failures
    remaining_boxes1_sorted, _ = torch.sort(remaining_boxes1, dim=0)
    small_boxes1_sorted, _ = torch.sort(small_boxes1, dim=0)
    assert torch.all(remaining_boxes1_sorted == small_boxes1_sorted)

    # In the second case, both boxes should be kept.
    remaining_boxes2 = filtered_priors_list[1].boxes.get(0)[0]
    expected_boxes2 = torch.cat([large_box2, small_boxes2])
    assert remaining_boxes2.shape[0] == 2
    # Sort boxes before comparison
    remaining_boxes2_sorted, _ = torch.sort(remaining_boxes2, dim=0)
    expected_boxes2_sorted, _ = torch.sort(expected_boxes2, dim=0)
    assert torch.all(remaining_boxes2_sorted == expected_boxes2_sorted)
