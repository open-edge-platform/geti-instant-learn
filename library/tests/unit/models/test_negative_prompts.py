# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for negative mask/point support across models."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from instantlearn.data.base.batch import Batch
from instantlearn.data.base.sample import BACKGROUND_CATEGORY_ID, Sample


def _make_ref_sample_with_background(
    img_size: tuple[int, int] = (224, 224),
) -> Sample:
    """Create a reference sample with one foreground and one background mask."""
    h, w = img_size
    fg_mask = np.zeros((h, w), dtype=np.uint8)
    fg_mask[50:100, 50:100] = 1  # foreground region

    bg_mask = np.zeros((h, w), dtype=np.uint8)
    bg_mask[150:200, 150:200] = 1  # background region

    return Sample(
        image=np.random.default_rng(42).integers(0, 255, (h, w, 3), dtype=np.uint8),
        masks=np.stack([fg_mask, bg_mask]),
        categories=["shoe", "background"],
        category_ids=np.array([1, BACKGROUND_CATEGORY_ID]),
        is_reference=[True, True],
    )


class TestMatcherNegativePrompts:
    """Test Matcher model negative prompt support."""

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_matcher_has_negative_mask_converter(
        self,
        mock_encoder: MagicMock,
        mock_sam: MagicMock,
    ) -> None:
        """Test that Matcher exposes the negative mask converter."""
        from instantlearn.components.negative_prompts import NegativeMaskToPoints
        from instantlearn.models.matcher import Matcher

        model = Matcher(device="cpu")
        assert hasattr(model, "negative_mask_converter")
        assert isinstance(model.negative_mask_converter, NegativeMaskToPoints)

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_matcher_num_negative_points_param(
        self,
        mock_encoder: MagicMock,
        mock_sam: MagicMock,
    ) -> None:
        """Test that num_negative_points is configurable."""
        from instantlearn.models.matcher import Matcher

        model = Matcher(device="cpu", num_negative_points=10)
        assert model.negative_mask_converter.num_points_per_mask == 10

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_matcher_extract_negative_embedding(
        self,
        mock_encoder: MagicMock,
        mock_sam: MagicMock,
    ) -> None:
        """Test that _extract_negative_embedding extracts features from background masks."""
        from instantlearn.models.matcher import Matcher

        mock_encoder_instance = MagicMock()
        mock_encoder_instance.input_size = 518
        mock_encoder_instance.patch_size = 14
        mock_encoder_instance.feature_size = 37
        mock_encoder.return_value = mock_encoder_instance

        model = Matcher(device="cpu", num_negative_points=3)

        ref = _make_ref_sample_with_background()
        batch = Batch.collate([ref])

        embed_dim = 64
        num_patches = 37 * 37
        fake_embeddings = torch.randn(1, num_patches, embed_dim)

        result = model._extract_negative_embedding(fake_embeddings, batch)

        assert result is not None
        assert result.shape == (1, embed_dim)
        # Should be L2-normalized
        assert torch.allclose(result.norm(dim=-1), torch.tensor(1.0), atol=1e-5)


class TestPerDinoNegativePrompts:
    """Test PerDino model negative prompt support."""

    @patch("instantlearn.models.per_dino.per_dino.load_sam_model")
    @patch("instantlearn.models.per_dino.per_dino.ImageEncoder")
    def test_per_dino_has_negative_mask_converter(
        self,
        mock_encoder: MagicMock,
        mock_sam: MagicMock,
    ) -> None:
        """Test that PerDino exposes the negative mask converter."""
        from instantlearn.components.negative_prompts import NegativeMaskToPoints
        from instantlearn.models.per_dino import PerDino

        model = PerDino(device="cpu")
        assert hasattr(model, "negative_mask_converter")
        assert isinstance(model.negative_mask_converter, NegativeMaskToPoints)

    @patch("instantlearn.models.per_dino.per_dino.load_sam_model")
    @patch("instantlearn.models.per_dino.per_dino.ImageEncoder")
    def test_per_dino_extract_negative_embedding(
        self,
        mock_encoder: MagicMock,
        mock_sam: MagicMock,
    ) -> None:
        """Test that _extract_negative_embedding extracts features from background masks."""
        from instantlearn.models.per_dino import PerDino

        mock_encoder_instance = MagicMock()
        mock_encoder_instance.input_size = 518
        mock_encoder_instance.patch_size = 14
        mock_encoder_instance.feature_size = 37
        mock_encoder.return_value = mock_encoder_instance

        model = PerDino(device="cpu", num_negative_points=4)

        ref = _make_ref_sample_with_background()
        batch = Batch.collate([ref])

        embed_dim = 64
        num_patches = 37 * 37
        fake_embeddings = torch.randn(1, num_patches, embed_dim)

        result = model._extract_negative_embedding(fake_embeddings, batch)

        assert result is not None
        assert result.shape == (1, embed_dim)
        # Should be L2-normalized
        assert torch.allclose(result.norm(dim=-1), torch.tensor(1.0), atol=1e-5)


class TestSoftMatcherNegativePrompts:
    """Test SoftMatcher inherits negative prompt support."""

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_soft_matcher_has_negative_mask_converter(
        self,
        mock_encoder: MagicMock,
        mock_sam: MagicMock,
    ) -> None:
        """Test that SoftMatcher inherits the negative mask converter."""
        from instantlearn.components.negative_prompts import NegativeMaskToPoints
        from instantlearn.models.soft_matcher import SoftMatcher

        model = SoftMatcher(device="cpu")
        assert hasattr(model, "negative_mask_converter")
        assert isinstance(model.negative_mask_converter, NegativeMaskToPoints)

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_soft_matcher_num_negative_points_passthrough(
        self,
        mock_encoder: MagicMock,
        mock_sam: MagicMock,
    ) -> None:
        """Test that SoftMatcher passes num_negative_points to Matcher."""
        from instantlearn.models.soft_matcher import SoftMatcher

        model = SoftMatcher(device="cpu", num_negative_points=8)
        assert model.negative_mask_converter.num_points_per_mask == 8


class TestGroundedSAMNegativePrompts:
    """Test GroundedSAM background category filtering."""

    @patch("instantlearn.models.grounded_sam.grounded_sam.load_sam_model")
    @patch("instantlearn.models.grounded_sam.grounded_sam.TextToBoxPromptGenerator")
    def test_grounded_sam_filters_background_category(
        self,
        mock_prompt_gen: MagicMock,
        mock_sam: MagicMock,
    ) -> None:
        """Test that fit() excludes background category from mapping."""
        from instantlearn.models.grounded_sam import GroundedSAM

        model = GroundedSAM(device="cpu")

        ref = Sample(
            image=np.random.default_rng(42).integers(0, 255, (64, 64, 3), dtype=np.uint8),
            categories=["shoe", "background", "bag"],
            category_ids=np.array([1, BACKGROUND_CATEGORY_ID, 2]),
        )
        model.fit(ref)

        assert "background" not in model.category_mapping
        assert "shoe" in model.category_mapping
        assert "bag" in model.category_mapping
        assert model.category_mapping["shoe"] == 1
        assert model.category_mapping["bag"] == 2

    @patch("instantlearn.models.grounded_sam.grounded_sam.load_sam_model")
    @patch("instantlearn.models.grounded_sam.grounded_sam.TextToBoxPromptGenerator")
    def test_grounded_sam_warns_on_background_masks(
        self,
        mock_prompt_gen: MagicMock,
        mock_sam: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that fit() logs a warning when background masks are provided."""
        import logging

        from instantlearn.models.grounded_sam import GroundedSAM

        model = GroundedSAM(device="cpu")

        ref = Sample(
            image=np.random.default_rng(42).integers(0, 255, (64, 64, 3), dtype=np.uint8),
            categories=["shoe", "background"],
            category_ids=np.array([1, BACKGROUND_CATEGORY_ID]),
        )
        with caplog.at_level(logging.WARNING):
            model.fit(ref)

        assert any("does not support negative prompts" in r.message for r in caplog.records)


class TestSAM3NegativePrompts:
    """Test SAM3 negative prompt support."""

    @patch("instantlearn.models.sam3.sam3.CLIPTokenizerFast")
    @patch("instantlearn.models.sam3.sam3.Sam3Model")
    def test_sam3_has_negative_mask_converter(
        self,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        """Test that SAM3 exposes the negative mask converter."""
        from instantlearn.components.negative_prompts import NegativeMaskToPoints
        from instantlearn.models.sam3.sam3 import SAM3

        model = SAM3(device="cpu")
        assert hasattr(model, "negative_mask_converter")
        assert isinstance(model.negative_mask_converter, NegativeMaskToPoints)
        assert model._negative_points is None

    @patch("instantlearn.models.sam3.sam3.CLIPTokenizerFast")
    @patch("instantlearn.models.sam3.sam3.Sam3Model")
    def test_sam3_extract_negative_points(
        self,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        """Test that fit() extracts negative points from background masks."""
        from instantlearn.models.sam3.sam3 import SAM3

        model = SAM3(device="cpu", num_negative_points=3)

        ref = _make_ref_sample_with_background()
        batch = Batch.collate([ref])

        model._extract_negative_points(batch)

        assert model._negative_points is not None
        assert model._negative_points.shape == (3, 2)
        # Points should be normalized to [0, 1]
        assert (model._negative_points >= 0).all()
        assert (model._negative_points <= 1).all()

    @patch("instantlearn.models.sam3.sam3.CLIPTokenizerFast")
    @patch("instantlearn.models.sam3.sam3.Sam3Model")
    def test_sam3_build_category_mapping_excludes_background(
        self,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        """Test that _build_category_mapping skips background category."""
        from instantlearn.models.sam3.sam3 import SAM3

        ref = _make_ref_sample_with_background()
        batch = Batch.collate([ref])

        mapping = SAM3._build_category_mapping(batch)

        assert "shoe" in mapping
        assert "background" not in mapping

    @patch("instantlearn.models.sam3.sam3.CLIPTokenizerFast")
    @patch("instantlearn.models.sam3.sam3.Sam3Model")
    def test_sam3_no_negative_points_without_background(
        self,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        """Test that _extract_negative_points produces nothing without background."""
        from instantlearn.models.sam3.sam3 import SAM3

        model = SAM3(device="cpu")

        ref = Sample(
            image=np.random.default_rng(42).integers(0, 255, (64, 64, 3), dtype=np.uint8),
            masks=np.ones((1, 64, 64), dtype=np.uint8),
            categories=["shoe"],
            category_ids=np.array([1]),
        )
        batch = Batch.collate([ref])

        model._extract_negative_points(batch)

        assert model._negative_points is None

    @patch("instantlearn.models.sam3.sam3.CLIPTokenizerFast")
    @patch("instantlearn.models.sam3.sam3.Sam3Model")
    def test_sam3_geometry_attrs_safe_without_background(
        self,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        """Test that _negative_geometry_features is None when no background masks exist.

        Verifies the __init__ initialization so that predict() can safely check
        ``if self._negative_geometry_features is not None`` without AttributeError.
        """
        from instantlearn.models.sam3.sam3 import SAM3

        model = SAM3(device="cpu")

        assert model._negative_geometry_features is None
        assert model._negative_geometry_mask is None

        # After fit() with no background, they should still be None
        ref = Sample(
            image=np.random.default_rng(42).integers(0, 255, (64, 64, 3), dtype=np.uint8),
            masks=np.ones((1, 64, 64), dtype=np.uint8),
            categories=["shoe"],
            category_ids=np.array([1]),
        )
        batch = Batch.collate([ref])
        model._extract_negative_points(batch)

        assert model._negative_geometry_features is None
        assert model._negative_geometry_mask is None
