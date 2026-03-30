# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for NMS post-processors."""

from __future__ import annotations

import pytest
import torch

from instantlearn.components.postprocessing.nms import (
    BoxIoMNMS,
    BoxNMS,
    MaskIoMNMS,
    MaskNMS,
    SoftNMS,
    _greedy_nms,  # noqa: PLC2701
    _pairwise_box_iom,  # noqa: PLC2701
    _pairwise_mask_iom,  # noqa: PLC2701
    _pairwise_mask_iou,  # noqa: PLC2701
)


def _make_mask(h: int, w: int, y1: int, x1: int, y2: int, x2: int) -> torch.Tensor:
    """Create a binary mask with a filled rectangle."""
    mask = torch.zeros(h, w, dtype=torch.bool)
    mask[y1:y2, x1:x2] = True
    return mask


class TestPairwiseMaskIoU:
    """Tests for pairwise mask IoU computation."""

    def test_identical_masks(self) -> None:
        """Test that identical masks produce IoU of 1.0."""
        masks = torch.ones(2, 10, 10, dtype=torch.bool)
        iou = _pairwise_mask_iou(masks)
        assert iou.shape == (2, 2)
        assert torch.allclose(iou, torch.ones(2, 2), atol=1e-4)

    def test_disjoint_masks(self) -> None:
        """Test that disjoint masks produce IoU of 0.0."""
        m1 = _make_mask(10, 10, 0, 0, 5, 5)
        m2 = _make_mask(10, 10, 5, 5, 10, 10)
        masks = torch.stack([m1, m2])
        iou = _pairwise_mask_iou(masks)
        assert torch.allclose(iou[0, 1], torch.tensor(0.0), atol=1e-4)
        assert torch.allclose(iou[1, 0], torch.tensor(0.0), atol=1e-4)

    def test_partial_overlap(self) -> None:
        """Test that partially overlapping masks produce expected IoU."""
        m1 = _make_mask(10, 10, 0, 0, 6, 6)
        m2 = _make_mask(10, 10, 3, 3, 9, 9)
        masks = torch.stack([m1, m2])
        iou = _pairwise_mask_iou(masks)
        # Intersection: 3x3=9, union: 36+36-9=63, IoU=9/63≈0.143
        assert 0.1 < iou[0, 1].item() < 0.2


class TestPairwiseMaskIoM:
    """Tests for pairwise mask IoM computation."""

    def test_contained_mask(self) -> None:
        """Test that a contained mask produces IoM of 1.0."""
        # m1 fully contained in m2
        m1 = _make_mask(10, 10, 2, 2, 4, 4)
        m2 = _make_mask(10, 10, 0, 0, 10, 10)
        masks = torch.stack([m1, m2])
        iom = _pairwise_mask_iom(masks)
        # Intersection = area(m1) = 4, min(4, 100) = 4, so IoM = 1.0
        assert torch.allclose(iom[0, 1], torch.tensor(1.0), atol=1e-4)

    def test_disjoint_masks(self) -> None:
        """Test that disjoint masks produce IoM of 0.0."""
        m1 = _make_mask(10, 10, 0, 0, 3, 3)
        m2 = _make_mask(10, 10, 7, 7, 10, 10)
        masks = torch.stack([m1, m2])
        iom = _pairwise_mask_iom(masks)
        assert torch.allclose(iom[0, 1], torch.tensor(0.0), atol=1e-4)


class TestPairwiseBoxIoM:
    """Tests for pairwise box IoM computation."""

    def test_contained_box(self) -> None:
        """Test that a contained box produces IoM of 1.0."""
        boxes = torch.tensor([
            [2.0, 2.0, 4.0, 4.0],  # small box
            [0.0, 0.0, 10.0, 10.0],  # large box
        ])
        iom = _pairwise_box_iom(boxes)
        assert torch.allclose(iom[0, 1], torch.tensor(1.0), atol=1e-4)

    def test_disjoint_boxes(self) -> None:
        """Test that disjoint boxes produce IoM of 0.0."""
        boxes = torch.tensor([
            [0.0, 0.0, 3.0, 3.0],
            [5.0, 5.0, 8.0, 8.0],
        ])
        iom = _pairwise_box_iom(boxes)
        assert torch.allclose(iom[0, 1], torch.tensor(0.0), atol=1e-4)


class TestGreedyNMS:
    """Tests for the greedy NMS algorithm."""

    def test_no_overlap_keeps_all(self) -> None:
        """Test that non-overlapping masks are all kept."""
        scores = torch.tensor([0.9, 0.8, 0.7])
        overlap = torch.zeros(3, 3)
        keep = _greedy_nms(scores, overlap, threshold=0.5)
        assert sorted(keep.tolist()) == [0, 1, 2]

    def test_high_overlap_suppresses_lower_score(self) -> None:
        """Test that highly overlapping detection with lower score is suppressed."""
        scores = torch.tensor([0.9, 0.8])
        overlap = torch.tensor([[1.0, 0.8], [0.8, 1.0]])
        keep = _greedy_nms(scores, overlap, threshold=0.5)
        assert keep.tolist() == [0]

    def test_containment_aware_prefers_larger_mask(self) -> None:
        """Containment-aware mode prefers the larger mask.

        When a small high-scored mask is contained inside a large lower-scored
        mask, the larger mask should be kept.
        """
        scores = torch.tensor([0.85, 0.80])  # idx=0 slightly higher
        overlap = torch.tensor([[1.0, 0.9], [0.9, 1.0]])
        areas = torch.tensor([100.0, 5000.0])  # idx=0 is tiny (contained)
        keep = _greedy_nms(
            scores, overlap, threshold=0.5,
            areas=areas, score_margin=0.1, area_ratio=0.5,
        )
        # idx=0 (small, high score) should be swapped out; idx=1 (large) kept
        assert 1 in keep.tolist()
        assert 0 not in keep.tolist()

    def test_single_mask_kept(self) -> None:
        """Test that a single mask is always kept."""
        scores = torch.tensor([0.5])
        overlap = torch.ones(1, 1)
        keep = _greedy_nms(scores, overlap, threshold=0.5)
        assert keep.tolist() == [0]


class TestMaskNMS:
    """Tests for MaskNMS post-processor."""

    def test_suppresses_overlapping_masks(self) -> None:
        """Test that overlapping masks are suppressed via IoU threshold."""
        m1 = _make_mask(10, 10, 0, 0, 8, 8)
        m2 = _make_mask(10, 10, 1, 1, 9, 9)  # heavily overlaps m1
        masks = torch.stack([m1, m2])
        scores = torch.tensor([0.9, 0.7])
        labels = torch.tensor([0, 0])

        nms = MaskNMS(iou_threshold=0.3)
        out_masks, out_scores, _ = nms(masks, scores, labels)
        assert out_masks.size(0) == 1
        assert out_scores[0].item() == pytest.approx(0.9)

    def test_keeps_disjoint_masks(self) -> None:
        """Test that disjoint masks are all preserved."""
        m1 = _make_mask(10, 10, 0, 0, 4, 4)
        m2 = _make_mask(10, 10, 6, 6, 10, 10)
        masks = torch.stack([m1, m2])
        scores = torch.tensor([0.9, 0.8])
        labels = torch.tensor([0, 1])

        nms = MaskNMS(iou_threshold=0.5)
        out_masks, _, _ = nms(masks, scores, labels)
        assert out_masks.size(0) == 2

    def test_single_mask_passthrough(self) -> None:
        """Test that a single mask passes through unchanged."""
        masks = torch.ones(1, 5, 5, dtype=torch.bool)
        scores = torch.tensor([0.5])
        labels = torch.tensor([0])

        nms = MaskNMS()
        out_masks, _, _ = nms(masks, scores, labels)
        assert out_masks.size(0) == 1


class TestBoxNMS:
    """Tests for BoxNMS post-processor."""

    def test_suppresses_overlapping_boxes(self) -> None:
        """Test that overlapping boxes are suppressed via torchvision NMS."""
        m1 = _make_mask(10, 10, 0, 0, 8, 8)
        m2 = _make_mask(10, 10, 1, 1, 9, 9)
        masks = torch.stack([m1, m2])
        scores = torch.tensor([0.9, 0.7])
        labels = torch.tensor([0, 0])

        nms = BoxNMS(iou_threshold=0.3)
        out_masks, _, _ = nms(masks, scores, labels)
        assert out_masks.size(0) == 1

    def test_single_mask_passthrough(self) -> None:
        """Test that a single mask passes through unchanged."""
        masks = torch.ones(1, 5, 5, dtype=torch.bool)
        scores = torch.tensor([0.5])
        labels = torch.tensor([0])

        nms = BoxNMS()
        out_masks, _, _ = nms(masks, scores, labels)
        assert out_masks.size(0) == 1


class TestMaskIoMNMS:
    """Tests for MaskIoMNMS post-processor."""
    def test_suppresses_contained_mask(self) -> None:
        """Small mask contained in a large one should be suppressed (IoM ≈ 1)."""
        m_large = _make_mask(20, 20, 0, 0, 20, 20)
        m_small = _make_mask(20, 20, 5, 5, 8, 8)
        masks = torch.stack([m_large, m_small])
        scores = torch.tensor([0.9, 0.8])
        labels = torch.tensor([0, 0])

        nms = MaskIoMNMS(iom_threshold=0.5)
        out_masks, out_scores, _ = nms(masks, scores, labels)
        assert out_masks.size(0) == 1
        assert out_scores[0].item() == pytest.approx(0.9)

    def test_keeps_non_overlapping(self) -> None:
        """Test that non-overlapping masks are preserved."""
        m1 = _make_mask(20, 20, 0, 0, 5, 5)
        m2 = _make_mask(20, 20, 15, 15, 20, 20)
        masks = torch.stack([m1, m2])
        scores = torch.tensor([0.9, 0.8])
        labels = torch.tensor([0, 1])

        nms = MaskIoMNMS(iom_threshold=0.3)
        out_masks, _, _ = nms(masks, scores, labels)
        assert out_masks.size(0) == 2


class TestBoxIoMNMS:
    """Tests for BoxIoMNMS post-processor."""

    def test_suppresses_contained_box(self) -> None:
        """Test that a contained box is suppressed."""
        m_large = _make_mask(20, 20, 0, 0, 20, 20)
        m_small = _make_mask(20, 20, 5, 5, 8, 8)
        masks = torch.stack([m_large, m_small])
        scores = torch.tensor([0.9, 0.8])
        labels = torch.tensor([0, 0])

        nms = BoxIoMNMS(iom_threshold=0.5)
        out_masks, _, _ = nms(masks, scores, labels)
        assert out_masks.size(0) == 1

    def test_containment_aware_mode(self) -> None:
        """Containment-aware mode should prefer the larger mask.

        When a small mask has slightly higher score, the larger mask should
        still be kept.
        """
        m_large = _make_mask(20, 20, 0, 0, 20, 20)
        m_small = _make_mask(20, 20, 8, 8, 10, 10)  # tiny, contained
        masks = torch.stack([m_small, m_large])
        scores = torch.tensor([0.85, 0.80])  # small has slightly higher score
        labels = torch.tensor([0, 0])

        nms = BoxIoMNMS(iom_threshold=0.3, score_margin=0.1, area_ratio=0.5)
        out_masks, out_scores, _ = nms(masks, scores, labels)
        assert out_masks.size(0) == 1
        # The large mask (idx=1) should be kept
        assert out_scores[0].item() == pytest.approx(0.80)


class TestSoftNMS:
    """Tests for SoftNMS post-processor."""

    def test_scores_decay_with_overlap(self) -> None:
        """Overlapping masks should have decayed scores."""
        m1 = _make_mask(10, 10, 0, 0, 8, 8)
        m2 = _make_mask(10, 10, 2, 2, 10, 10)  # significant overlap
        masks = torch.stack([m1, m2])
        scores = torch.tensor([0.9, 0.8])
        labels = torch.tensor([0, 0])

        soft_nms = SoftNMS(sigma=0.5, score_threshold=0.01)
        _, out_scores, _ = soft_nms(masks, scores, labels)
        # m2's score should decay due to overlap with m1
        assert out_scores[0].item() == pytest.approx(0.9)  # winner unchanged
        assert out_scores[1].item() < 0.8  # decayed

    def test_disjoint_no_decay(self) -> None:
        """Test that disjoint masks have no score decay."""
        m1 = _make_mask(10, 10, 0, 0, 4, 4)
        m2 = _make_mask(10, 10, 6, 6, 10, 10)
        masks = torch.stack([m1, m2])
        scores = torch.tensor([0.9, 0.8])
        labels = torch.tensor([0, 1])

        soft_nms = SoftNMS(sigma=0.5, score_threshold=0.01)
        out_masks, out_scores, _ = soft_nms(masks, scores, labels)
        assert out_masks.size(0) == 2
        assert torch.allclose(out_scores, scores, atol=1e-4)

    def test_removes_below_threshold(self) -> None:
        """Heavily overlapping masks should be removed if score decays below threshold."""
        m1 = _make_mask(10, 10, 0, 0, 10, 10)
        m2 = _make_mask(10, 10, 0, 0, 10, 10)  # identical
        masks = torch.stack([m1, m2])
        scores = torch.tensor([0.9, 0.15])
        labels = torch.tensor([0, 0])

        soft_nms = SoftNMS(sigma=0.1, score_threshold=0.1)
        out_masks, _, _ = soft_nms(masks, scores, labels)
        # m2 should be removed due to massive decay with IoU=1.0
        assert out_masks.size(0) == 1

    def test_single_mask_passthrough(self) -> None:
        """Test that a single mask passes through SoftNMS unchanged."""
        masks = torch.ones(1, 5, 5, dtype=torch.bool)
        scores = torch.tensor([0.5])
        labels = torch.tensor([0])

        soft_nms = SoftNMS()
        out_masks, _, _ = soft_nms(masks, scores, labels)
        assert out_masks.size(0) == 1
