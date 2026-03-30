# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for PostProcessorPipeline."""

from __future__ import annotations

import torch
from torch import nn

from instantlearn.components.postprocessing.base import PostProcessor, PostProcessorPipeline


class _DoubleScores(PostProcessor):
    """Test processor that doubles scores."""

    def forward(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return masks, scores * 2, labels


class _DropFirst(PostProcessor):
    """Test processor that drops the first mask."""

    def forward(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return masks[1:], scores[1:], labels[1:]


class TestPostProcessorPipeline:
    """Tests for PostProcessorPipeline chaining behavior."""

    def test_chains_processors_sequentially(self) -> None:
        """Output of one processor feeds into the next."""
        masks = torch.ones(3, 5, 5, dtype=torch.bool)
        scores = torch.tensor([0.5, 0.3, 0.2])
        labels = torch.tensor([0, 1, 2])

        pipeline = PostProcessorPipeline([_DropFirst(), _DoubleScores()])
        out_masks, out_scores, _ = pipeline(masks, scores, labels)

        # First processor drops first mask (3 → 2), second doubles scores
        assert out_masks.size(0) == 2
        assert torch.allclose(out_scores, torch.tensor([0.6, 0.4]))

    def test_empty_pipeline_passthrough(self) -> None:
        """Test that an empty pipeline passes data through unchanged."""
        masks = torch.ones(2, 5, 5, dtype=torch.bool)
        scores = torch.tensor([0.9, 0.8])
        labels = torch.tensor([0, 1])

        pipeline = PostProcessorPipeline([])
        out_masks, out_scores, _ = pipeline(masks, scores, labels)

        assert out_masks.size(0) == 2
        assert torch.equal(out_scores, scores)

    def test_len(self) -> None:
        """Test that len returns number of processors in pipeline."""
        pipeline = PostProcessorPipeline([_DoubleScores(), _DropFirst()])
        assert len(pipeline) == 2

    def test_empty_input(self) -> None:
        """Test that empty input produces empty output."""
        masks = torch.empty(0, 5, 5, dtype=torch.bool)
        scores = torch.empty(0)
        labels = torch.empty(0, dtype=torch.int64)

        pipeline = PostProcessorPipeline([_DoubleScores()])
        out_masks, _, _ = pipeline(masks, scores, labels)
        assert out_masks.size(0) == 0

    def test_is_nn_module(self) -> None:
        """Pipeline should be a proper nn.Module for ONNX export compatibility."""
        pipeline = PostProcessorPipeline([_DoubleScores()])
        assert isinstance(pipeline, nn.Module)
        # Submodules should be registered
        submodules = list(pipeline.modules())
        assert len(submodules) > 1
