# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ONNX-export prompt-selection paths of PerDino and SoftMatcher.

These validate the static-shape export branches added for the backend-agnostic
migration: ``GridPromptGenerator`` (PerDino) and ``SoftmatcherPromptGenerator``
(SoftMatcher). The eager paths use data-dependent ops (``torch.where`` /
``nonzero`` / ``multinomial`` / RFF) that are unsafe for ONNX; the export paths
must be static-shaped and deterministic while preserving the scoring math.
"""

from unittest.mock import patch

import torch
from torch.nn import functional

from instantlearn.models.per_dino.prompt_generators import GridPromptGenerator
from instantlearn.models.soft_matcher.prompt_generator import SoftmatcherPromptGenerator


class TestGridPromptExport:
    """GridPromptGenerator export branch (PerDino)."""

    def _gen(self) -> GridPromptGenerator:
        return GridPromptGenerator(
            num_grid_cells=2,
            point_selection_threshold=0.5,
            num_bg_points=2,
            num_foreground_points=5,
            max_points=7,
        )

    def test_export_static_shape(self) -> None:
        """Under ONNX export the output is a fixed [T, C, max_points, 4] tensor."""
        gen = self._gen()
        similarities = torch.rand(1, 2, 8, 8)
        original_sizes = torch.tensor([[64, 64]])
        with patch("torch.onnx.is_in_onnx_export", return_value=True):
            out = gen(similarities, [1, 2], original_sizes)
        assert out.shape == (1, 2, 7, 4)

    def test_export_is_deterministic(self) -> None:
        """The export path uses only top-K (no sampling), so it is deterministic."""
        gen = self._gen()
        similarities = torch.rand(1, 1, 8, 8)
        original_sizes = torch.tensor([[64, 64]])
        with patch("torch.onnx.is_in_onnx_export", return_value=True):
            a = gen(similarities, [1], original_sizes)
            b = gen(similarities, [1], original_sizes)
        assert torch.equal(a, b)

    def test_export_foreground_picks_argmax(self) -> None:
        """The top foreground point is the similarity-map argmax."""
        gen = GridPromptGenerator(num_bg_points=0, num_foreground_points=1, max_points=1)
        sim = torch.zeros(1, 1, 8, 8)
        sim[0, 0, 3, 5] = 1.0  # peak at row=3, col=5
        original_sizes = torch.tensor([[8, 8]])  # map==image, no scaling
        with patch("torch.onnx.is_in_onnx_export", return_value=True):
            out = gen(sim, [1], original_sizes)
        x, y, _score, label = out[0, 0, 0].tolist()
        assert (x, y) == (5.0, 3.0)
        assert label == 1.0


class TestSoftMatcherPromptExport:
    """SoftmatcherPromptGenerator export branch (SoftMatcher)."""

    def _gen(self, **kwargs: object) -> SoftmatcherPromptGenerator:
        return SoftmatcherPromptGenerator(
            encoder_input_size=224,
            encoder_patch_size=14,
            encoder_feature_size=16,
            num_foreground_points=kwargs.pop("num_foreground_points", 5),
            num_background_points=kwargs.pop("num_background_points", 2),
            **kwargs,
        )

    def test_export_soft_scores_match_aggregation(self) -> None:
        """The exported foreground point equals the soft-correspondence argmax."""
        torch.manual_seed(0)
        feat, embed = 16, 8
        num_patches = feat * feat
        ref_embed = torch.randn(num_patches, embed)
        target_embed = torch.randn(num_patches, embed)
        masked_ref_embed = ref_embed.mean(dim=0)
        mask = torch.zeros(num_patches)
        mask[:10] = 1.0
        original_size = torch.tensor([224, 224])  # scale == 1

        gen = self._gen(num_foreground_points=1, num_background_points=0, softmatching_bidirectional=False)
        with patch("torch.onnx.is_in_onnx_export", return_value=True):
            points, similarity = gen._process_single_category_export(  # noqa: SLF001
                ref_embed, masked_ref_embed, mask, target_embed, original_size,
            )

        # Recompute the soft-correspondence scores (unidirectional).
        sim_map = ref_embed @ target_embed.T
        log_fwd = functional.log_softmax(sim_map / 0.1, dim=1)
        log_corr = (log_fwd * mask.unsqueeze(1)).sum(dim=0) / mask.sum()
        scores = torch.exp(log_corr)
        best = int(scores.argmax())
        exp_col, exp_row = best % feat, best // feat
        exp_x = exp_col * 14 + 7
        exp_y = exp_row * 14 + 7

        x, y, _s, label = points[0].tolist()
        assert (x, y) == (exp_x, exp_y)
        assert label == 1.0
        assert similarity.shape == (feat, feat)

    def test_export_ignores_sampling_and_rff(self) -> None:
        """Export stays deterministic even with sampling/RFF flags enabled."""
        torch.manual_seed(0)
        feat, embed = 16, 8
        num_patches = feat * feat
        ref = torch.randn(num_patches, embed)
        tgt = torch.randn(num_patches, embed)
        masked = ref.mean(dim=0)
        mask = torch.zeros(num_patches)
        mask[:10] = 1.0
        original_size = torch.tensor([224, 224])

        gen = self._gen(
            use_sampling=True,
            use_spatial_sampling=True,
            approximate_matching=True,
        )
        with patch("torch.onnx.is_in_onnx_export", return_value=True):
            a, _ = gen._process_single_category_export(ref, masked, mask, tgt, original_size)  # noqa: SLF001
            b, _ = gen._process_single_category_export(ref, masked, mask, tgt, original_size)  # noqa: SLF001
        assert torch.equal(a, b)
