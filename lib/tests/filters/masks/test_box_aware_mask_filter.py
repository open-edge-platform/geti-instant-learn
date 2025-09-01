# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test the BoxAwareMaskFilter."""


import pytest
import torch
from getiprompt.filters.masks.box_aware_mask_filter import BoxAwareMaskFilter
from getiprompt.types.boxes import Boxes
from getiprompt.types.masks import Masks


@pytest.fixture
def filter_instance() -> BoxAwareMaskFilter:
    """Returns an instance of the BoxAwareMaskFilter."""
    return BoxAwareMaskFilter()


class TestBoxAwareMaskFilter:
    """Test cases for the BoxAwareMaskFilter."""

    @staticmethod
    @pytest.mark.parametrize(
        ("input_masks_tensors", "expected_masks_tensors"),
        [
            (
                [
                    torch.tensor([[True, True, False], [True, True, False], [False, False, False]]),
                    torch.tensor([[True, True, True], [True, True, True], [False, False, False]]),
                ],
                [
                    torch.tensor([[True, True, True], [True, True, True], [False, False, False]]),
                ],
            ),
            (
                [
                    torch.tensor([[True, True, False], [True, True, False], [False, False, False]]),
                    torch.tensor([[False, False, True], [False, False, True], [False, False, True]]),
                ],
                [
                    torch.tensor([[True, True, False], [True, True, False], [False, False, False]]),
                    torch.tensor([[False, False, True], [False, False, True], [False, False, True]]),
                ],
            ),
            (
                [
                    torch.tensor([[True, True, False], [True, True, False], [False, False, False]]),
                    torch.tensor([[False, True, True], [False, True, True], [False, False, False]]),
                ],
                [
                    torch.tensor([[True, True, False], [True, True, False], [False, False, False]]),
                    torch.tensor([[False, True, True], [False, True, True], [False, False, False]]),
                ],
            ),
            (
                [
                    torch.tensor([[True, True], [True, False]]),
                    torch.tensor([[True, True], [True, False]]),
                ],
                [
                    torch.tensor([[True, True], [True, False]]),
                ],
            ),
            (
                [
                    torch.tensor([[False, False], [False, False]]),
                    torch.tensor([[True, True], [True, False]]),
                ],
                [
                    torch.tensor([[True, True], [True, False]]),
                ],
            ),
            (
                [
                    torch.tensor([[True, True, True, True], [True, True, True, True]]),
                    torch.tensor([[True, True, True, True], [True, True, True, False]]),
                ],
                [
                    torch.tensor([[True, True, True, True], [True, True, True, True]]),
                ],
            ),
        ],
    )
    def test_filtering_scenarios(
        filter_instance: BoxAwareMaskFilter,
        input_masks_tensors: list[torch.Tensor],
        expected_masks_tensors: list[torch.Tensor],
    ) -> None:
        """Test various mask filtering scenarios."""
        masks = Masks()
        # Add batch dimension and stack
        stacked_masks = torch.stack([m.unsqueeze(0) for m in input_masks_tensors], dim=0).squeeze(1)
        masks.data[0] = stacked_masks

        filtered_masks, _ = filter_instance([masks])

        if 0 in filtered_masks[0].data and filtered_masks[0].data[0].numel() > 0:
            result_masks = filtered_masks[0].data[0]
            assert result_masks.shape[0] == len(expected_masks_tensors)

            # Check that the filtered masks are the expected ones, order-independent
            result_masks_list = list(result_masks)
            expected_masks_list = list(torch.stack(expected_masks_tensors))

            for expected_mask in expected_masks_list:
                assert any(torch.equal(expected_mask, res_mask) for res_mask in result_masks_list)

        else:
            assert len(expected_masks_tensors) == 0

    @staticmethod
    def test_mask_box_correspondence(filter_instance: BoxAwareMaskFilter) -> None:
        """Test that boxes are filtered to match filtered masks."""
        # Create test masks
        masks = Masks()
        small_mask = torch.tensor([[True, False], [False, False]]).unsqueeze(0)
        large_mask = torch.tensor([[True, True], [True, False]]).unsqueeze(0)  # Contains small_mask
        stacked_masks = torch.cat([small_mask, large_mask], dim=0)
        masks.data[0] = stacked_masks

        # Create corresponding boxes
        boxes = Boxes()
        box_tensor = torch.tensor([
            [0, 0, 1, 1, 0.9, 1],  # Box for small mask (should be removed)
            [0, 0, 2, 2, 0.8, 1],  # Box for large mask (should be kept)
        ])
        boxes.add(box_tensor, class_id=0)

        filtered_masks, filtered_boxes = filter_instance([masks], [boxes])

        # Should have 1 mask and 1 corresponding box
        assert filtered_masks[0].data[0].shape[0] == 1
        assert torch.equal(filtered_masks[0].data[0][0], large_mask.squeeze(0))
        assert len(filtered_boxes[0].get(0)) == 1
        assert filtered_boxes[0].get(0)[0].shape[0] == 1  # One box remaining

    @staticmethod
    def test_multiple_classes(filter_instance: BoxAwareMaskFilter) -> None:
        """Test filtering works correctly with multiple classes."""
        masks = Masks()

        # Class 0: small mask contained in large mask
        small_mask = torch.tensor([[True, False], [False, False]]).unsqueeze(0)
        large_mask = torch.tensor([[True, True], [True, False]]).unsqueeze(0)
        masks.data[0] = torch.cat([small_mask, large_mask], dim=0)

        # Class 1: two non-overlapping masks
        mask1 = torch.tensor([[True, False], [False, False]]).unsqueeze(0)
        mask2 = torch.tensor([[False, False], [False, True]]).unsqueeze(0)
        masks.data[1] = torch.cat([mask1, mask2], dim=0)

        filtered_masks, _ = filter_instance([masks])

        # Class 0 should have 1 mask (large one), Class 1 should have 2 masks
        assert filtered_masks[0].data[0].shape[0] == 1  # Class 0
        assert torch.equal(filtered_masks[0].data[0][0], large_mask.squeeze(0))
        assert filtered_masks[0].data[1].shape[0] == 2  # Class 1

    @staticmethod
    def test_multiple_images(filter_instance: BoxAwareMaskFilter) -> None:
        """Test filtering works correctly with multiple images."""
        # Image 1
        masks1 = Masks()
        small_mask = torch.tensor([[True, False], [False, False]]).unsqueeze(0)
        large_mask = torch.tensor([[True, True], [True, False]]).unsqueeze(0)
        masks1.data[0] = torch.cat([small_mask, large_mask], dim=0)

        # Image 2
        masks2 = Masks()
        mask_a = torch.tensor([[True, False], [False, False]]).unsqueeze(0)
        mask_b = torch.tensor([[False, False], [False, True]]).unsqueeze(0)
        masks2.data[0] = torch.cat([mask_a, mask_b], dim=0)

        filtered_masks, _ = filter_instance([masks1, masks2])

        # Image 1 should have 1 mask, Image 2 should have 2 masks
        expected_image_count = 2
        expected_mask_count_image_1 = 1
        expected_mask_count_image_2 = 2
        assert len(filtered_masks) == expected_image_count
        assert filtered_masks[0].data[0].shape[0] == expected_mask_count_image_1  # Image 1
        assert torch.equal(filtered_masks[0].data[0][0], large_mask.squeeze(0))
        assert filtered_masks[1].data[0].shape[0] == expected_mask_count_image_2  # Image 2

    @staticmethod
    def test_empty_masks(filter_instance: BoxAwareMaskFilter) -> None:
        """Test handling of empty masks."""
        masks = Masks()
        # Create empty mask data
        empty_tensor = torch.empty((0, 3, 3), dtype=torch.bool)
        masks.data[0] = empty_tensor

        filtered_masks, _ = filter_instance([masks])

        # Should handle empty masks gracefully
        assert len(filtered_masks) == 1
        assert 0 not in filtered_masks[0].data or filtered_masks[0].data[0].shape[0] == 0

    @staticmethod
    def test_no_boxes_provided(filter_instance: BoxAwareMaskFilter) -> None:
        """Test that filter works when no boxes are provided."""
        masks = Masks()
        small_mask = torch.tensor([[True, False], [False, False]]).unsqueeze(0)
        large_mask = torch.tensor([[True, True], [True, False]]).unsqueeze(0)
        masks.data[0] = torch.cat([small_mask, large_mask], dim=0)

        filtered_masks, filtered_boxes = filter_instance([masks])

        # Should return filtered masks and None for boxes
        assert filtered_masks[0].data[0].shape[0] == 1
        assert torch.equal(filtered_masks[0].data[0][0], large_mask.squeeze(0))
        assert filtered_boxes is None
