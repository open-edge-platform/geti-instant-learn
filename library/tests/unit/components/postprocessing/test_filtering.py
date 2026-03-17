# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for filtering post-processors."""

from __future__ import annotations

import pytest
import torch

from instantlearn.components.postprocessing.filtering import MinimumAreaFilter, ScoreFilter


class TestScoreFilter:
    """Tests for ScoreFilter post-processor."""

    def test_removes_zero_scores(self) -> None:
        """Test that zero-scored masks are filtered out."""
        masks = torch.ones(3, 5, 5, dtype=torch.bool)
        scores = torch.tensor([0.8, 0.0, 0.5])
        labels = torch.tensor([0, 1, 2])

        filt = ScoreFilter(min_score=0.0)
        out_masks, out_scores, _ = filt(masks, scores, labels)
        assert out_masks.size(0) == 2
        assert 0.0 not in out_scores.tolist()

    def test_custom_threshold(self) -> None:
        """Test that custom min_score threshold filters correctly."""
        masks = torch.ones(4, 5, 5, dtype=torch.bool)
        scores = torch.tensor([0.9, 0.3, 0.5, 0.1])
        labels = torch.tensor([0, 1, 2, 3])

        filt = ScoreFilter(min_score=0.4)
        out_masks, out_scores, _ = filt(masks, scores, labels)
        assert out_masks.size(0) == 2
        assert all(s > 0.4 for s in out_scores.tolist())

    def test_empty_input(self) -> None:
        """Test that empty input returns empty output."""
        masks = torch.empty(0, 5, 5, dtype=torch.bool)
        scores = torch.empty(0)
        labels = torch.empty(0, dtype=torch.int64)

        filt = ScoreFilter()
        out_masks, _, _ = filt(masks, scores, labels)
        assert out_masks.size(0) == 0

    def test_all_pass(self) -> None:
        """Test that all masks pass when all scores exceed threshold."""
        masks = torch.ones(3, 5, 5, dtype=torch.bool)
        scores = torch.tensor([0.9, 0.8, 0.7])
        labels = torch.tensor([0, 1, 2])

        filt = ScoreFilter(min_score=0.0)
        out_masks, _, _ = filt(masks, scores, labels)
        assert out_masks.size(0) == 3


class TestMinimumAreaFilter:
    """Tests for MinimumAreaFilter post-processor."""

    def test_removes_small_masks(self) -> None:
        """Test that masks below min_area are removed."""
        m1 = torch.ones(10, 10, dtype=torch.bool)  # area = 100
        m2 = torch.zeros(10, 10, dtype=torch.bool)
        m2[0, 0] = True  # area = 1
        masks = torch.stack([m1, m2])
        scores = torch.tensor([0.9, 0.8])
        labels = torch.tensor([0, 1])

        filt = MinimumAreaFilter(min_area=50)
        out_masks, out_scores, _ = filt(masks, scores, labels)
        assert out_masks.size(0) == 1
        assert out_scores[0].item() == pytest.approx(0.9)

    def test_empty_input(self) -> None:
        """Test that empty input returns empty output."""
        masks = torch.empty(0, 5, 5, dtype=torch.bool)
        scores = torch.empty(0)
        labels = torch.empty(0, dtype=torch.int64)

        filt = MinimumAreaFilter()
        out_masks, _, _ = filt(masks, scores, labels)
        assert out_masks.size(0) == 0

    def test_all_above_threshold(self) -> None:
        """Test that all masks pass when above min_area threshold."""
        masks = torch.ones(2, 10, 10, dtype=torch.bool)
        scores = torch.tensor([0.9, 0.8])
        labels = torch.tensor([0, 1])

        filt = MinimumAreaFilter(min_area=10)
        out_masks, _, _ = filt(masks, scores, labels)
        assert out_masks.size(0) == 2

    def test_boundary_area(self) -> None:
        """Mask at exactly min_area should be kept (>= threshold)."""
        mask = torch.zeros(10, 10, dtype=torch.bool)
        mask[0:5, 0:2] = True  # area = 10
        masks = mask.unsqueeze(0)
        scores = torch.tensor([0.5])
        labels = torch.tensor([0])

        filt = MinimumAreaFilter(min_area=10)
        out_masks, _, _ = filt(masks, scores, labels)
        assert out_masks.size(0) == 1
