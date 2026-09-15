# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from instantlearn.models.matcher.prompt_generators import BidirectionalPromptGenerator


def _make_generator(
    num_foreground: int = 10,
    num_background: int = 2,
    feat_size: int = 8,
    patch_size: int = 14,
    input_size: int = 112,
) -> BidirectionalPromptGenerator:
    return BidirectionalPromptGenerator(
        encoder_input_size=input_size,
        encoder_patch_size=patch_size,
        encoder_feature_size=feat_size,
        num_foreground_points=num_foreground,
        num_background_points=num_background,
    )


class TestSelectBackgroundPointsZeroMask:
    """_select_background_points must not produce NaN when ref_mask is all-zero."""

    def test_empty_ref_mask_returns_no_nan_avg_similarity(self) -> None:
        """avg_similarity is all-zeros (not NaN) when no foreground patches exist."""
        gen = _make_generator(num_background=2, feat_size=8)
        num_ref = 64
        num_target = 64
        similarity_map = torch.randn(num_ref, num_target)
        ref_mask = torch.zeros(num_ref)  # zero-coverage: no foreground patches

        avg_sim, bg_idx, bg_scores = gen._select_background_points(similarity_map, ref_mask)

        assert not torch.isnan(avg_sim).any(), "avg_similarity must not contain NaN for zero-coverage mask"
        assert not torch.isnan(bg_scores).any(), "bg_scores must not contain NaN for zero-coverage mask"

    def test_empty_ref_mask_returns_k_background_indices(self) -> None:
        """Exactly num_background_points indices are returned even for an all-zero mask."""
        gen = _make_generator(num_background=2, feat_size=8)
        num_ref = 64
        num_target = 64
        similarity_map = torch.randn(num_ref, num_target)
        ref_mask = torch.zeros(num_ref)

        _, bg_idx, bg_scores = gen._select_background_points(similarity_map, ref_mask)

        assert bg_idx.numel() == 2
        assert bg_scores.numel() == 2

    def test_empty_ref_mask_indices_within_bounds(self) -> None:
        """All returned background indices are valid target-patch indices."""
        gen = _make_generator(num_background=3, feat_size=8)
        num_ref = 64
        num_target = 64
        similarity_map = torch.randn(num_ref, num_target)
        ref_mask = torch.zeros(num_ref)

        _, bg_idx, _ = gen._select_background_points(similarity_map, ref_mask)

        assert (bg_idx >= 0).all()
        assert (bg_idx < num_target).all()

    def test_non_empty_ref_mask_behaviour_unchanged(self) -> None:
        """The fix must not alter behaviour when the ref_mask has foreground patches."""
        gen = _make_generator(num_background=2, feat_size=8)
        torch.manual_seed(0)
        num_ref = 64
        num_target = 64
        similarity_map = torch.randn(num_ref, num_target)
        ref_mask = torch.zeros(num_ref)
        ref_mask[:10] = 1.0  # some foreground patches

        avg_sim, bg_idx, bg_scores = gen._select_background_points(similarity_map, ref_mask)

        assert not torch.isnan(avg_sim).any()
        assert bg_idx.numel() == 2
        assert not torch.isnan(bg_scores).any()


class TestProcessSingleCategoryZeroCoverage:
    """_process_single_category must return valid (non-NaN, non-crash) padded points
    when the reference mask covers zero encoder patch cells."""

    def _run(self, feat_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        gen = _make_generator(num_foreground=5, num_background=2, feat_size=feat_size)
        num_patches = feat_size * feat_size
        embed_dim = 64

        torch.manual_seed(42)
        ref_embed = torch.randn(num_patches, embed_dim)
        # Zero masked embedding — what MaskedFeatureExtractor returns for zero-coverage.
        masked_ref_embed = torch.zeros(1, embed_dim)
        # All-zero flatten_ref_mask — no foreground patches.
        flatten_ref_mask = torch.zeros(num_patches)
        target_embed = torch.randn(num_patches, embed_dim)
        original_size = torch.tensor([feat_size * 14, feat_size * 14])  # H, W

        padded_points, similarity = gen._process_single_category(
            ref_embed,
            masked_ref_embed,
            flatten_ref_mask,
            target_embed,
            original_size,
        )
        return padded_points, similarity

    def test_no_crash(self) -> None:
        """Must complete without raising any exception."""
        self._run()  # should not raise

    def test_output_shapes(self) -> None:
        """padded_points has shape [max_points, 4]; similarity has shape [feat_size, feat_size]."""
        gen = _make_generator(num_foreground=5, num_background=2, feat_size=8)
        padded_points, similarity = self._run(feat_size=8)

        assert padded_points.shape == (gen.max_points, 4)
        assert similarity.shape == (8, 8)

    def test_no_nan_in_padded_points(self) -> None:
        """padded_points must not contain NaN — NaN coords would crash SAM downstream."""
        padded_points, _ = self._run()
        assert not torch.isnan(padded_points).any(), "padded_points must not contain NaN for zero-coverage mask"

    def test_no_nan_in_similarity(self) -> None:
        """similarity must not contain NaN."""
        _, similarity = self._run()
        assert not torch.isnan(similarity).any()

    def test_all_foreground_labels_are_zero(self) -> None:
        """With zero-coverage mask there are no foreground matches; all non-padded points
        should have label 0 (padded) or -1 (background).  Crucially, label=1 (foreground)
        must be absent because there were no matched foreground patches."""
        padded_points, _ = self._run()
        labels = padded_points[:, 3]
        assert not (labels == 1).any(), (
            "Zero-coverage category must produce no foreground (label=1) points"
        )

    @pytest.mark.parametrize("feat_size", [4, 8, 16])
    def test_various_feat_sizes(self, feat_size: int) -> None:
        """Output shapes are correct across different feature-grid sizes."""
        gen = _make_generator(num_foreground=5, num_background=2, feat_size=feat_size)
        padded_points, similarity = self._run(feat_size=feat_size)

        assert padded_points.shape == (gen.max_points, 4)
        assert similarity.shape == (feat_size, feat_size)
        assert not torch.isnan(padded_points).any()


class TestPromptGeneratorForwardZeroCoverage:
    """BidirectionalPromptGenerator.forward must handle batches where one category
    has an all-zero flatten_ref_mask and a zero masked_ref_embedding."""

    def test_forward_two_categories_one_zero_coverage(self) -> None:
        """forward must not crash and must return correctly shaped tensors."""
        feat_size = 8
        num_patches = feat_size * feat_size
        embed_dim = 64
        num_targets = 1
        num_categories = 2

        gen = _make_generator(num_foreground=5, num_background=2, feat_size=feat_size)

        torch.manual_seed(7)
        # Category 0 – normal coverage
        ref_embeds_cat0 = torch.randn(num_patches, embed_dim)
        masked_embed_cat0 = torch.randn(1, embed_dim)
        masked_embed_cat0 = masked_embed_cat0 / masked_embed_cat0.norm(dim=-1, keepdim=True)
        mask_cat0 = (torch.rand(num_patches) > 0.5).float()

        # Category 1 – zero coverage (tiny annotation)
        ref_embeds_cat1 = torch.randn(num_patches, embed_dim)
        masked_embed_cat1 = torch.zeros(1, embed_dim)  # zero embedding
        mask_cat1 = torch.zeros(num_patches)            # all-zero mask

        # Stack into [C, ...] tensors
        ref_embeddings = torch.stack([ref_embeds_cat0, ref_embeds_cat1])   # [2, N, D]
        masked_ref_embeddings = torch.stack([masked_embed_cat0, masked_embed_cat1])  # [2, 1, D]
        flatten_ref_masks = torch.stack([mask_cat0, mask_cat1])             # [2, N]

        target_embeddings = torch.randn(num_targets, num_patches, embed_dim)
        original_sizes = torch.tensor([[feat_size * 14, feat_size * 14]])
        category_ids = [1, 2]

        point_prompts, similarities = gen.forward(
            ref_embeddings,
            masked_ref_embeddings,
            flatten_ref_masks,
            category_ids,
            target_embeddings,
            original_sizes,
        )

        assert point_prompts.shape == (num_targets, num_categories, gen.max_points, 4)
        assert similarities.shape == (num_targets, num_categories, feat_size, feat_size)
        assert not torch.isnan(point_prompts).any(), "point_prompts must not contain NaN"
        assert not torch.isnan(similarities).any(), "similarities must not contain NaN"

    def test_forward_all_zero_coverage(self) -> None:
        """Edge case: every category has zero coverage. Must complete without crash."""
        feat_size = 8
        num_patches = feat_size * feat_size
        embed_dim = 32

        gen = _make_generator(num_foreground=3, num_background=1, feat_size=feat_size)

        ref_embeddings = torch.randn(2, num_patches, embed_dim)
        masked_ref_embeddings = torch.zeros(2, 1, embed_dim)
        flatten_ref_masks = torch.zeros(2, num_patches)
        target_embeddings = torch.randn(1, num_patches, embed_dim)
        original_sizes = torch.tensor([[112, 112]])
        category_ids = [5, 9]

        point_prompts, similarities = gen.forward(
            ref_embeddings,
            masked_ref_embeddings,
            flatten_ref_masks,
            category_ids,
            target_embeddings,
            original_sizes,
        )

        assert point_prompts.shape == (1, 2, gen.max_points, 4)
        assert not torch.isnan(point_prompts).any()
