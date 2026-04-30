# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SAM3/EfficientSAM3 post-processing functions."""

from __future__ import annotations

import pytest
import torch

from instantlearn.models.sam3.post_processing import (
    PostProcessingConfig,
    apply_non_overlapping_constraint,
    apply_post_processing,
    box_nms,
    mask_iom_suppression,
)


@pytest.fixture
def overlapping_boxes() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two highly overlapping boxes and one separate box.

    Box 0 and Box 1 overlap ~68% IoU. Box 2 is far away.
    """
    scores = torch.tensor([0.9, 0.7, 0.5])
    boxes = torch.tensor([
        [10.0, 10.0, 110.0, 110.0],  # 100x100 box
        [20.0, 20.0, 120.0, 120.0],  # 100x100 box, shifted 10px
        [300.0, 300.0, 400.0, 400.0],  # separate box
    ])
    masks = torch.zeros(3, 200, 200, dtype=torch.int64)
    masks[0, 10:110, 10:110] = 1
    masks[1, 20:120, 20:120] = 1
    masks[2, 150:200, 150:200] = 1
    return scores, boxes, masks


@pytest.fixture
def overlapping_masks() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three masks where mask 0 and 1 heavily overlap, mask 2 is separate."""
    scores = torch.tensor([0.9, 0.8, 0.5])
    boxes = torch.tensor([
        [10.0, 10.0, 60.0, 60.0],
        [10.0, 10.0, 55.0, 55.0],  # almost fully contained in mask 0
        [80.0, 80.0, 100.0, 100.0],
    ])
    masks = torch.zeros(3, 100, 100, dtype=torch.int64)
    masks[0, 10:60, 10:60] = 1  # 50x50
    masks[1, 10:55, 10:55] = 1  # 45x45, contained in mask 0
    masks[2, 80:100, 80:100] = 1  # separate
    return scores, boxes, masks


# PostProcessingConfig
class TestPostProcessingConfig:
    """Tests for PostProcessingConfig defaults and construction."""

    def test_defaults(self) -> None:
        """Verify all defaults are disabled."""
        config = PostProcessingConfig()
        assert config.nms_iou_threshold is None
        assert config.mask_iom_threshold is None
        assert config.non_overlapping_masks is False

    def test_custom_values(self) -> None:
        """Verify custom values are stored correctly."""
        config = PostProcessingConfig(
            nms_iou_threshold=0.5,
            mask_iom_threshold=0.3,
            non_overlapping_masks=True,
        )
        assert config.nms_iou_threshold == pytest.approx(0.5)
        assert config.mask_iom_threshold == pytest.approx(0.3)
        assert config.non_overlapping_masks is True


# box_nms


class TestBoxNMS:
    """Tests for torchvision-based box NMS."""

    def test_suppresses_overlapping(
        self,
        overlapping_boxes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """With IoU threshold 0.5, overlapping box 1 should be suppressed."""
        scores, boxes, masks = overlapping_boxes
        out_scores, _, _ = box_nms(scores, boxes, masks, iou_threshold=0.5)

        assert len(out_scores) == 2
        # Highest-scored (0.9) and non-overlapping (0.5) survive
        assert torch.allclose(out_scores, torch.tensor([0.9, 0.5]))

    def test_keeps_all_with_high_threshold(
        self,
        overlapping_boxes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """With threshold 1.0 (nothing suppressed), all boxes survive."""
        scores, boxes, masks = overlapping_boxes
        out_scores, _, _ = box_nms(scores, boxes, masks, iou_threshold=1.0)
        assert len(out_scores) == 3

    def test_empty_input(self) -> None:
        """Empty tensors pass through unchanged."""
        scores = torch.tensor([])
        boxes = torch.empty(0, 4)
        masks = torch.empty(0, 10, 10, dtype=torch.int64)
        out_s, out_b, out_m = box_nms(scores, boxes, masks, iou_threshold=0.5)
        assert out_s.numel() == 0
        assert out_b.shape == (0, 4)
        assert out_m.shape == (0, 10, 10)

    def test_single_prediction(self) -> None:
        """Single prediction always survives."""
        scores = torch.tensor([0.8])
        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        masks = torch.ones(1, 20, 20, dtype=torch.int64)
        out_s, _, _ = box_nms(scores, boxes, masks, iou_threshold=0.5)
        assert len(out_s) == 1


class TestMaskIoMSuppression:
    """Tests for greedy IoM-based mask overlap removal."""

    def test_suppresses_contained_mask(
        self,
        overlapping_masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Mask 1 is almost fully contained in mask 0 → should be suppressed."""
        scores, boxes, masks = overlapping_masks
        out_scores, _, _ = mask_iom_suppression(
            scores,
            boxes,
            masks,
            iom_threshold=0.3,
        )

        # Mask 1 (IoM ≈ 1.0 against mask 0) is suppressed
        assert len(out_scores) == 2
        assert torch.allclose(out_scores, torch.tensor([0.9, 0.5]))

    def test_keeps_all_with_high_threshold(
        self,
        overlapping_masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """With threshold 1.0, nothing is suppressed."""
        scores, boxes, masks = overlapping_masks
        out_scores, _, _ = mask_iom_suppression(
            scores,
            boxes,
            masks,
            iom_threshold=1.0,
        )
        assert len(out_scores) == 3

    def test_empty_input(self) -> None:
        """Empty tensors pass through unchanged."""
        scores = torch.tensor([])
        boxes = torch.empty(0, 4)
        masks = torch.empty(0, 10, 10, dtype=torch.int64)
        out_s, _, _ = mask_iom_suppression(
            scores,
            boxes,
            masks,
            iom_threshold=0.3,
        )
        assert out_s.numel() == 0

    def test_single_prediction(self) -> None:
        """Single prediction always survives."""
        scores = torch.tensor([0.8])
        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        masks = torch.ones(1, 20, 20, dtype=torch.int64)
        out_s, _, _ = mask_iom_suppression(scores, boxes, masks, iom_threshold=0.3)
        assert len(out_s) == 1

    def test_no_overlap(self) -> None:
        """Completely separate masks should all survive."""
        scores = torch.tensor([0.9, 0.7])
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [90.0, 90.0, 100.0, 100.0]])
        masks = torch.zeros(2, 100, 100, dtype=torch.int64)
        masks[0, :10, :10] = 1
        masks[1, 90:100, 90:100] = 1
        out_s, _, _ = mask_iom_suppression(scores, boxes, masks, iom_threshold=0.3)
        assert len(out_s) == 2


class TestNonOverlappingConstraint:
    """Tests for pixel-level non-overlapping constraint."""

    def test_winner_takes_all(self) -> None:
        """At overlapping pixels, only the highest-scored mask survives."""
        scores = torch.tensor([0.9, 0.6])
        masks = torch.zeros(2, 20, 20, dtype=torch.int64)
        masks[0, 5:15, 5:15] = 1  # 10x10
        masks[1, 8:18, 8:18] = 1  # 10x10, overlaps 7x7 with mask 0

        result = apply_non_overlapping_constraint(scores, masks)

        # Overlapping region (8:15, 8:15) should belong to mask 0 only
        overlap_region_0 = result[0, 8:15, 8:15]
        overlap_region_1 = result[1, 8:15, 8:15]
        assert overlap_region_0.all(), "Higher-scored mask should keep overlap pixels"
        assert not overlap_region_1.any(), "Lower-scored mask should lose overlap pixels"

        # Non-overlapping regions should be preserved
        assert result[0, 5:8, 5:8].all()  # mask 0's unique region
        assert result[1, 15:18, 15:18].all()  # mask 1's unique region

    def test_no_overlap_unchanged(self) -> None:
        """Non-overlapping masks should be unchanged."""
        scores = torch.tensor([0.9, 0.5])
        masks = torch.zeros(2, 20, 20, dtype=torch.int64)
        masks[0, 0:5, 0:5] = 1
        masks[1, 15:20, 15:20] = 1

        result = apply_non_overlapping_constraint(scores, masks)
        assert torch.equal(result, masks)

    def test_single_mask_unchanged(self) -> None:
        """Single mask is unchanged."""
        masks = torch.ones(1, 10, 10, dtype=torch.int64)
        scores = torch.tensor([0.9])
        result = apply_non_overlapping_constraint(scores, masks)
        assert torch.equal(result, masks)

    def test_empty_input(self) -> None:
        """Empty masks pass through."""
        masks = torch.empty(0, 10, 10, dtype=torch.int64)
        scores = torch.tensor([])
        result = apply_non_overlapping_constraint(scores, masks)
        assert result.shape == (0, 10, 10)


class TestApplyPostProcessing:
    """Tests for the orchestrator function."""

    def test_none_config_passthrough(
        self,
        overlapping_boxes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """With None config, inputs are returned unchanged."""
        scores, boxes, masks = overlapping_boxes
        out_s, out_b, out_m = apply_post_processing(scores, boxes, masks, config=None)
        assert torch.equal(out_s, scores)
        assert torch.equal(out_b, boxes)
        assert torch.equal(out_m, masks)

    def test_all_disabled_passthrough(
        self,
        overlapping_boxes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """With all algorithms disabled, inputs are returned unchanged."""
        scores, boxes, masks = overlapping_boxes
        config = PostProcessingConfig()  # all defaults = disabled
        out_s, out_b, out_m = apply_post_processing(scores, boxes, masks, config)
        assert torch.equal(out_s, scores)
        assert torch.equal(out_b, boxes)
        assert torch.equal(out_m, masks)

    def test_nms_only(
        self,
        overlapping_boxes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Only box NMS enabled."""
        config = PostProcessingConfig(nms_iou_threshold=0.5)
        scores, boxes, masks = overlapping_boxes
        out_s, _, _ = apply_post_processing(scores, boxes, masks, config)
        assert len(out_s) == 2

    def test_iom_only(
        self,
        overlapping_masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Only mask IoM suppression enabled."""
        config = PostProcessingConfig(mask_iom_threshold=0.3)
        scores, boxes, masks = overlapping_masks
        out_s, _, _ = apply_post_processing(scores, boxes, masks, config)
        assert len(out_s) == 2

    def test_non_overlapping_only(self) -> None:
        """Only non-overlapping pixel constraint enabled."""
        config = PostProcessingConfig(non_overlapping_masks=True)
        scores = torch.tensor([0.9, 0.6])
        boxes = torch.tensor([[5.0, 5.0, 15.0, 15.0], [8.0, 8.0, 18.0, 18.0]])
        masks = torch.zeros(2, 20, 20, dtype=torch.int64)
        masks[0, 5:15, 5:15] = 1
        masks[1, 8:18, 8:18] = 1

        _, _, out_m = apply_post_processing(scores, boxes, masks, config)
        # At overlap (8:15, 8:15), only mask 0 should survive
        assert out_m[0, 8:15, 8:15].all()
        assert not out_m[1, 8:15, 8:15].any()

    def test_combined_nms_and_iom(
        self,
        overlapping_masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Both NMS and IoM enabled — contained mask is removed by at least one."""
        config = PostProcessingConfig(nms_iou_threshold=0.5, mask_iom_threshold=0.3)
        scores, boxes, masks = overlapping_masks
        out_s, _, _ = apply_post_processing(scores, boxes, masks, config)
        assert len(out_s) == 2

    def test_empty_input(self) -> None:
        """Empty inputs with config pass through."""
        config = PostProcessingConfig(nms_iou_threshold=0.5)
        scores = torch.tensor([])
        boxes = torch.empty(0, 4)
        masks = torch.empty(0, 10, 10, dtype=torch.int64)
        out_s, _, _ = apply_post_processing(scores, boxes, masks, config)
        assert out_s.numel() == 0
