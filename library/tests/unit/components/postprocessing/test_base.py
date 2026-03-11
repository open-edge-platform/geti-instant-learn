# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for apply_postprocessing helper."""

from __future__ import annotations

import torch

from instantlearn.components.postprocessing.base import (
    PostProcessor,
    apply_postprocessing,
)


class _DropLowScore(PostProcessor):
    """Test processor that drops masks with score < 0.5."""

    def forward(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keep = scores >= 0.5
        return masks[keep], scores[keep], labels[keep]


class _DecayScores(PostProcessor):
    """Test processor that halves all scores without changing masks."""

    def forward(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return masks, scores * 0.5, labels


class _Identity(PostProcessor):
    """Test processor that returns inputs unchanged."""

    def forward(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return masks, scores, labels


def _make_predictions() -> list[dict[str, torch.Tensor]]:
    """Create a sample prediction dict."""
    masks = torch.zeros(3, 10, 10, dtype=torch.bool)
    masks[0, 0:5, 0:5] = True
    masks[1, 2:8, 2:8] = True
    masks[2, 7:10, 7:10] = True
    scores = torch.tensor([0.9, 0.3, 0.7])
    labels = torch.tensor([0, 1, 2])

    # Manually create pred_boxes: [x1, y1, x2, y2, score]
    pred_boxes = torch.tensor([
        [0.0, 0.0, 5.0, 5.0, 0.9],
        [2.0, 2.0, 8.0, 8.0, 0.3],
        [7.0, 7.0, 10.0, 10.0, 0.7],
    ])

    return [{"pred_masks": masks, "pred_scores": scores, "pred_labels": labels, "pred_boxes": pred_boxes}]


class TestApplyPostprocessing:
    """Tests for the apply_postprocessing helper function."""

    def test_none_postprocessor_returns_as_is(self) -> None:
        """Test that None postprocessor returns predictions unchanged."""
        preds = _make_predictions()
        result = apply_postprocessing(preds, None)
        assert result is preds

    def test_masks_change_recomputes_boxes(self) -> None:
        """When masks change (filtering), pred_boxes should be recomputed."""
        preds = _make_predictions()
        processor = _DropLowScore()
        result = apply_postprocessing(preds, processor)

        assert result[0]["pred_masks"].size(0) == 2  # dropped score=0.3
        assert result[0]["pred_scores"].size(0) == 2
        assert result[0]["pred_labels"].size(0) == 2
        assert result[0]["pred_boxes"].shape == (2, 5)

    def test_scores_change_updates_box_scores(self) -> None:
        """When only scores change (e.g. SoftNMS decay), pred_boxes score column should be refreshed.

        The pred_boxes score column (column 4) should match the new scores.
        """
        preds = _make_predictions()
        processor = _DecayScores()
        result = apply_postprocessing(preds, processor)

        # Masks unchanged, but scores halved
        assert result[0]["pred_masks"].size(0) == 3
        expected_scores = torch.tensor([0.45, 0.15, 0.35])
        assert torch.allclose(result[0]["pred_scores"], expected_scores, atol=1e-4)
        # pred_boxes score column (column 4) should match new scores
        assert torch.allclose(result[0]["pred_boxes"][:, 4], expected_scores, atol=1e-4)

    def test_preserves_extra_keys(self) -> None:
        """Test that extra prediction keys are preserved through processing."""
        preds = _make_predictions()
        preds[0]["pred_points"] = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        processor = _Identity()
        result = apply_postprocessing(preds, processor)
        assert "pred_points" in result[0]

    def test_empty_masks_after_processing(self) -> None:
        """All masks filtered out should produce empty pred_boxes."""
        masks = torch.zeros(1, 5, 5, dtype=torch.bool)
        masks[0, 0, 0] = True
        scores = torch.tensor([0.1])  # below threshold
        labels = torch.tensor([0])
        preds = [{"pred_masks": masks, "pred_scores": scores, "pred_labels": labels}]

        processor = _DropLowScore()
        result = apply_postprocessing(preds, processor)

        assert result[0]["pred_masks"].size(0) == 0
        assert result[0]["pred_boxes"].shape == (0, 5)

    def test_multiple_predictions_in_batch(self) -> None:
        """Test that batch processing works across multiple prediction dicts."""
        preds = _make_predictions() + _make_predictions()
        processor = _DropLowScore()
        result = apply_postprocessing(preds, processor)
        assert len(result) == 2
        for r in result:
            assert r["pred_masks"].size(0) == 2

    def test_identity_preserves_original_boxes(self) -> None:
        """When masks and scores are unchanged, original pred_boxes should be preserved."""
        preds = _make_predictions()
        original_boxes = preds[0]["pred_boxes"].clone()
        processor = _Identity()
        result = apply_postprocessing(preds, processor)
        assert torch.equal(result[0]["pred_boxes"], original_boxes)

    def test_boxes_computed_when_missing_from_input(self) -> None:
        """Point-prompt models (Matcher/PerDino) don't produce pred_boxes; they should be derived from masks."""
        masks = torch.zeros(2, 10, 10, dtype=torch.bool)
        masks[0, 1:4, 2:6] = True
        masks[1, 5:9, 3:8] = True
        scores = torch.tensor([0.9, 0.8])
        labels = torch.tensor([0, 1])
        preds = [{"pred_masks": masks, "pred_scores": scores, "pred_labels": labels}]

        processor = _Identity()
        result = apply_postprocessing(preds, processor)

        assert "pred_boxes" in result[0]
        assert result[0]["pred_boxes"].shape == (2, 5)
        # Verify box coordinates match mask extents
        assert torch.allclose(result[0]["pred_boxes"][0, :4], torch.tensor([2.0, 1.0, 5.0, 3.0]))
        assert torch.allclose(result[0]["pred_boxes"][1, :4], torch.tensor([3.0, 5.0, 7.0, 8.0]))

    def test_boxes_computed_empty_when_no_masks_and_missing_from_input(self) -> None:
        """When no masks remain and pred_boxes was never in the input, output should have empty boxes."""
        masks = torch.zeros(1, 5, 5, dtype=torch.bool)
        masks[0, 0, 0] = True
        scores = torch.tensor([0.1])
        labels = torch.tensor([0])
        preds = [{"pred_masks": masks, "pred_scores": scores, "pred_labels": labels}]

        processor = _DropLowScore()
        result = apply_postprocessing(preds, processor)

        assert "pred_boxes" in result[0]
        assert result[0]["pred_boxes"].shape == (0, 5)
