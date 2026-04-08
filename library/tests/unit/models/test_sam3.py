# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SAM3 model (classic and visual exemplar modes).

These tests mock the underlying Sam3Model, tokenizer, and preprocessors
to validate SAM3 logic without loading real weights.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from instantlearn.data.base.batch import Batch
from instantlearn.data.base.sample import Sample
from instantlearn.models.sam3.sam3 import SAM3, Sam3PromptMode

# -- Helpers --


def _make_mock_model() -> MagicMock:
    """Create a mock Sam3Model with plausible return values."""
    model = MagicMock()
    model.eval.return_value = model
    model.to.return_value = model

    hidden = 256
    num_queries = 200
    h_feat, w_feat = 63, 63  # 1008 / 16

    # Vision features returned by get_vision_features
    fpn = [torch.randn(1, hidden, h_feat * 2, w_feat * 2) for _ in range(3)]
    fpn.append(torch.randn(1, hidden, h_feat, w_feat))
    fpn_pos = [torch.randn_like(f) for f in fpn]
    model.get_vision_features.return_value = {
        "fpn_hidden_states": fpn,
        "fpn_position_encoding": fpn_pos,
    }

    # Text features
    text_output = MagicMock()
    text_output.pooler_output = torch.randn(1, 32, hidden)
    model.get_text_features.return_value = text_output

    # Geometry encoder
    model.geometry_encoder.return_value = {
        "last_hidden_state": torch.randn(1, 2, hidden),
        "attention_mask": torch.ones(1, 2, dtype=torch.bool),
    }

    # Forward pass returns (detection output)
    model.return_value = {
        "pred_logits": torch.randn(1, num_queries),
        "pred_boxes": torch.rand(1, num_queries, 4),
        "pred_masks": torch.randn(1, num_queries, h_feat, w_feat),
        "presence_logits": torch.randn(1, 1),
    }

    return model


def _make_mock_tokenizer() -> MagicMock:
    """Create a mock CLIPTokenizerFast."""
    tokenizer = MagicMock()
    tokenizer.return_value = MagicMock(
        input_ids=torch.ones(1, 32, dtype=torch.long),
        attention_mask=torch.ones(1, 32, dtype=torch.long),
    )
    return tokenizer


def _make_mock_postprocessor() -> MagicMock:
    """Create a mock Sam3Postprocessor."""
    post = MagicMock()
    post.to.return_value = post
    # Return one result per image with 3 detections
    post.return_value = [
        {
            "scores": torch.tensor([0.9, 0.8, 0.7]),
            "boxes": torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100], [110, 110, 150, 150]], dtype=torch.float32),
            "masks": torch.ones(3, 224, 224, dtype=torch.int64),
        },
    ]
    return post


def _make_mock_preprocessor() -> MagicMock:
    """Create a mock Sam3Preprocessor."""
    pre = MagicMock()
    pre.to.return_value = pre
    pre.return_value = (
        torch.randn(1, 3, 1008, 1008),
        torch.tensor([[224, 224]], dtype=torch.int32),
    )
    return pre


def _make_mock_prompt_preprocessor() -> MagicMock:
    """Create a mock Sam3PromptPreprocessor."""
    pre = MagicMock()
    pre.to.return_value = pre
    # Return (normalized_boxes, normalized_points)
    pre.return_value = (
        torch.tensor([[[0.3, 0.3, 0.1, 0.1]]]),  # cxcywh
        torch.tensor([[[0.5, 0.5]]]),  # xy
    )
    return pre


@pytest.fixture
def mock_sam3_deps() -> dict[str, Any]:
    """Create all mocked SAM3 dependencies."""
    return {
        "model": _make_mock_model(),
        "tokenizer": _make_mock_tokenizer(),
        "preprocessor": _make_mock_preprocessor(),
        "prompt_preprocessor": _make_mock_prompt_preprocessor(),
        "postprocessor": _make_mock_postprocessor(),
    }


def _build_sam3(mock_deps: dict[str, Any], prompt_mode: Sam3PromptMode = Sam3PromptMode.CLASSIC) -> SAM3:
    """Build an SAM3 instance with mocked internals."""
    with (
        patch("instantlearn.models.sam3.sam3.Sam3Model") as mock_model_cls,
        patch("instantlearn.models.sam3.sam3.CLIPTokenizerFast") as mock_tok_cls,
        patch("instantlearn.models.sam3.sam3.Sam3Preprocessor") as mock_pre_cls,
        patch("instantlearn.models.sam3.sam3.Sam3PromptPreprocessor") as mock_ppre_cls,
        patch("instantlearn.models.sam3.sam3.Sam3Postprocessor") as mock_post_cls,
    ):
        mock_model_cls.from_pretrained.return_value = mock_deps["model"]
        mock_tok_cls.from_pretrained.return_value = mock_deps["tokenizer"]
        mock_pre_cls.return_value = mock_deps["preprocessor"]
        mock_ppre_cls.return_value = mock_deps["prompt_preprocessor"]
        mock_post_cls.return_value = mock_deps["postprocessor"]

        return SAM3(device="cpu", precision="fp32", prompt_mode=prompt_mode)


# ---------------------------------------------------------------------------
# Tests: SAM3 initialization
# ---------------------------------------------------------------------------


class TestSAM3Initialization:
    """Test SAM3 initialization for both prompt modes."""

    @pytest.mark.parametrize("prompt_mode", [Sam3PromptMode.CLASSIC, Sam3PromptMode.VISUAL_EXEMPLAR])
    def test_initialization(self, mock_sam3_deps: dict[str, Any], prompt_mode: Sam3PromptMode) -> None:
        """Test SAM3 initializes with correct attributes."""
        model = _build_sam3(mock_sam3_deps, prompt_mode)

        assert model.prompt_mode == prompt_mode
        assert model.device == "cpu"
        assert model.precision == "fp32"
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert callable(model.fit)
        assert callable(model.predict)

    def test_initialization_string_prompt_mode(self, mock_sam3_deps: dict[str, Any]) -> None:
        """Test SAM3 accepts string prompt_mode."""
        with (
            patch("instantlearn.models.sam3.sam3.Sam3Model") as mock_model_cls,
            patch("instantlearn.models.sam3.sam3.CLIPTokenizerFast") as mock_tok_cls,
            patch("instantlearn.models.sam3.sam3.Sam3Preprocessor") as mock_pre_cls,
            patch("instantlearn.models.sam3.sam3.Sam3PromptPreprocessor") as mock_ppre_cls,
            patch("instantlearn.models.sam3.sam3.Sam3Postprocessor") as mock_post_cls,
        ):
            mock_model_cls.from_pretrained.return_value = mock_sam3_deps["model"]
            mock_tok_cls.from_pretrained.return_value = mock_sam3_deps["tokenizer"]
            mock_pre_cls.return_value = mock_sam3_deps["preprocessor"]
            mock_ppre_cls.return_value = mock_sam3_deps["prompt_preprocessor"]
            mock_post_cls.return_value = mock_sam3_deps["postprocessor"]

            model = SAM3(device="cpu", precision="fp32", prompt_mode="visual_exemplar")

        assert model.prompt_mode == Sam3PromptMode.VISUAL_EXEMPLAR

    def test_default_prompt_mode_is_classic(self, mock_sam3_deps: dict[str, Any]) -> None:
        """Test SAM3 defaults to classic prompt mode."""
        with (
            patch("instantlearn.models.sam3.sam3.Sam3Model") as mock_model_cls,
            patch("instantlearn.models.sam3.sam3.CLIPTokenizerFast") as mock_tok_cls,
            patch("instantlearn.models.sam3.sam3.Sam3Preprocessor") as mock_pre_cls,
            patch("instantlearn.models.sam3.sam3.Sam3PromptPreprocessor") as mock_ppre_cls,
            patch("instantlearn.models.sam3.sam3.Sam3Postprocessor") as mock_post_cls,
        ):
            mock_model_cls.from_pretrained.return_value = mock_sam3_deps["model"]
            mock_tok_cls.from_pretrained.return_value = mock_sam3_deps["tokenizer"]
            mock_pre_cls.return_value = mock_sam3_deps["preprocessor"]
            mock_ppre_cls.return_value = mock_sam3_deps["prompt_preprocessor"]
            mock_post_cls.return_value = mock_sam3_deps["postprocessor"]

            model = SAM3(device="cpu", precision="fp32")

        assert model.prompt_mode == Sam3PromptMode.CLASSIC

    def test_exemplar_state_initially_none(self, mock_sam3_deps: dict[str, Any]) -> None:
        """Test exemplar features start as None."""
        model = _build_sam3(mock_sam3_deps, Sam3PromptMode.VISUAL_EXEMPLAR)

        assert model.exemplar_geometry_features is None
        assert model.exemplar_geometry_mask is None
        assert model.exemplar_text_features is None
        assert model.exemplar_text_mask is None
        assert model.exemplar_category_ids is None


# ---------------------------------------------------------------------------
# Tests: Classic mode fit / predict
# ---------------------------------------------------------------------------


class TestSAM3Classic:
    """Test SAM3 classic prompt mode."""

    def test_fit_stores_category_mapping(self, mock_sam3_deps: dict[str, Any]) -> None:
        """Test fit() in classic mode stores category mapping."""
        model = _build_sam3(mock_sam3_deps, Sam3PromptMode.CLASSIC)

        ref = Sample(categories=["shoe", "bag"], category_ids=[0, 1])
        model.fit(ref)

        assert model.category_mapping is not None
        assert model.category_mapping == {"shoe": 0, "bag": 1}

    def test_fit_multiple_samples(self, mock_sam3_deps: dict[str, Any]) -> None:
        """Test fit() merges categories across samples."""
        model = _build_sam3(mock_sam3_deps, Sam3PromptMode.CLASSIC)

        refs = [
            Sample(categories=["shoe"], category_ids=[0]),
            Sample(categories=["bag"], category_ids=[1]),
        ]
        model.fit(refs)

        assert model.category_mapping == {"shoe": 0, "bag": 1}

    def test_predict_returns_correct_structure(self, mock_sam3_deps: dict[str, Any]) -> None:
        """Test predict() in classic mode returns expected keys and shapes."""
        model = _build_sam3(mock_sam3_deps, Sam3PromptMode.CLASSIC)

        ref = Sample(categories=["shoe"], category_ids=[0])
        model.fit(ref)

        target = Sample(image=torch.zeros(3, 224, 224))
        predictions = model.predict(target)

        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert "pred_masks" in predictions[0]
        assert "pred_boxes" in predictions[0]
        assert "pred_labels" in predictions[0]
        assert isinstance(predictions[0]["pred_masks"], torch.Tensor)

    def test_predict_mask_spatial_shape(self, mock_sam3_deps: dict[str, Any]) -> None:
        """Test that masks match target image spatial dims."""
        model = _build_sam3(mock_sam3_deps, Sam3PromptMode.CLASSIC)

        ref = Sample(categories=["shoe"], category_ids=[0])
        model.fit(ref)

        target = Sample(image=torch.zeros(3, 224, 224))
        predictions = model.predict(target)

        masks = predictions[0]["pred_masks"]
        if masks.numel() > 0:
            assert masks.shape[-2:] == (224, 224)

    def test_predict_multiple_targets(self, mock_sam3_deps: dict[str, Any]) -> None:
        """Test predict() with multiple target images."""
        model = _build_sam3(mock_sam3_deps, Sam3PromptMode.CLASSIC)

        ref = Sample(categories=["shoe"], category_ids=[0])
        model.fit(ref)

        targets = [
            Sample(image=torch.zeros(3, 224, 224)),
            Sample(image=torch.zeros(3, 320, 320)),
        ]
        predictions = model.predict(targets)

        assert len(predictions) == 2
        for pred in predictions:
            assert "pred_masks" in pred
            assert "pred_boxes" in pred
            assert "pred_labels" in pred

    def test_predict_with_bbox_prompts(self, mock_sam3_deps: dict[str, Any]) -> None:
        """Test classic predict with bounding box prompts on target."""
        model = _build_sam3(mock_sam3_deps, Sam3PromptMode.CLASSIC)

        target = Sample(
            image=torch.zeros(3, 224, 224),
            bboxes=np.array([[10, 10, 50, 50]]),
            categories=["shoe"],
            category_ids=[0],
        )
        predictions = model.predict(target)

        assert isinstance(predictions, list)
        assert len(predictions) == 1


# ---------------------------------------------------------------------------
# Tests: Visual exemplar mode fit / predict
# ---------------------------------------------------------------------------


class TestSAM3VisualExemplar:
    """Test SAM3 visual exemplar prompt mode."""

    def test_fit_caches_exemplar_features(self, mock_sam3_deps: dict[str, Any]) -> None:
        """Test fit() in visual mode caches geometry + text features."""
        model = _build_sam3(mock_sam3_deps, Sam3PromptMode.VISUAL_EXEMPLAR)

        ref = Sample(
            image=torch.zeros(3, 224, 224),
            bboxes=np.array([[10, 10, 50, 50]]),
            categories=["shoe"],
            category_ids=np.array([0]),
        )
        model.fit(ref)

        assert model.exemplar_geometry_features is not None
        assert model.exemplar_geometry_mask is not None
        assert model.exemplar_text_features is not None
        assert model.exemplar_text_mask is not None
        assert model.exemplar_category_ids is not None
        assert 0 in model.exemplar_category_ids

    def test_fit_with_points(self, mock_sam3_deps: dict[str, Any]) -> None:
        """Test fit() in visual mode with point prompts."""
        model = _build_sam3(mock_sam3_deps, Sam3PromptMode.VISUAL_EXEMPLAR)

        ref = Sample(
            image=torch.zeros(3, 224, 224),
            points=np.array([[100, 100]]),
            categories=["shoe"],
            category_ids=np.array([0]),
        )
        model.fit(ref)

        assert model.exemplar_geometry_features is not None
        assert model.exemplar_category_ids is not None

    def test_fit_raises_without_prompts(self, mock_sam3_deps: dict[str, Any]) -> None:
        """Test fit() raises ValueError without bboxes or points."""
        model = _build_sam3(mock_sam3_deps, Sam3PromptMode.VISUAL_EXEMPLAR)

        ref = Sample(
            image=torch.zeros(3, 224, 224),
            categories=["shoe"],
            category_ids=[0],
        )

        with pytest.raises(ValueError, match="bboxes or points"):
            model.fit(ref)

    def test_fit_raises_without_image(self, mock_sam3_deps: dict[str, Any]) -> None:
        """Test fit() raises ValueError when prompts provided but no image."""
        model = _build_sam3(mock_sam3_deps, Sam3PromptMode.VISUAL_EXEMPLAR)

        ref = Sample(
            bboxes=np.array([[10, 10, 50, 50]]),
            categories=["shoe"],
            category_ids=np.array([0]),
        )

        with pytest.raises(ValueError, match="images"):
            model.fit(ref)

    def test_fit_stores_category_mapping(self, mock_sam3_deps: dict[str, Any]) -> None:
        """Test fit() in visual mode also stores category mapping."""
        model = _build_sam3(mock_sam3_deps, Sam3PromptMode.VISUAL_EXEMPLAR)

        ref = Sample(
            image=torch.zeros(3, 224, 224),
            bboxes=np.array([[10, 10, 50, 50]]),
            categories=["shoe"],
            category_ids=np.array([0]),
        )
        model.fit(ref)

        assert model.category_mapping is not None
        assert "shoe" in model.category_mapping

    def test_fit_multi_category(self, mock_sam3_deps: dict[str, Any]) -> None:
        """Test fit() with multiple categories across samples."""
        model = _build_sam3(mock_sam3_deps, Sam3PromptMode.VISUAL_EXEMPLAR)

        refs = [
            Sample(
                image=torch.zeros(3, 224, 224),
                bboxes=np.array([[10, 10, 50, 50]]),
                categories=["shoe"],
                category_ids=np.array([0]),
            ),
            Sample(
                image=torch.zeros(3, 224, 224),
                bboxes=np.array([[60, 60, 100, 100]]),
                categories=["bag"],
                category_ids=np.array([1]),
            ),
        ]
        model.fit(refs)

        assert len(model.exemplar_category_ids) == 2
        assert set(model.exemplar_category_ids) == {0, 1}

    def test_predict_returns_correct_structure(self, mock_sam3_deps: dict[str, Any]) -> None:
        """Test predict() in visual mode returns expected keys."""
        model = _build_sam3(mock_sam3_deps, Sam3PromptMode.VISUAL_EXEMPLAR)

        ref = Sample(
            image=torch.zeros(3, 224, 224),
            bboxes=np.array([[10, 10, 50, 50]]),
            categories=["shoe"],
            category_ids=np.array([0]),
        )
        model.fit(ref)

        target = Sample(image=torch.zeros(3, 224, 224))
        predictions = model.predict(target)

        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert "pred_masks" in predictions[0]
        assert "pred_boxes" in predictions[0]
        assert "pred_labels" in predictions[0]

    def test_fit_with_long_category_name(self, mock_sam3_deps: dict[str, Any]) -> None:
        """Test fit() truncates long category names (e.g. UUIDs) to CLIP max_length.

        The backend passes label UUIDs as category names, which tokenize to ~35
        subword tokens — exceeding CLIP's 32-token max_position_embeddings.
        The tokenizer must truncate to avoid ValueError.
        """
        model = _build_sam3(mock_sam3_deps, Sam3PromptMode.VISUAL_EXEMPLAR)

        uuid_category = "550e8400-e29b-41d4-a716-446655440000"
        ref = Sample(
            image=torch.zeros(3, 224, 224),
            bboxes=np.array([[10, 10, 50, 50]]),
            categories=[uuid_category],
            category_ids=np.array([0]),
        )
        model.fit(ref)

        # Tokenizer must have been called with truncation=True
        call_args = mock_sam3_deps["tokenizer"].call_args
        assert call_args is not None
        _, kwargs = call_args
        assert kwargs.get("truncation") is True

        assert model.exemplar_text_features is not None

    def test_predict_raises_without_fit(self, mock_sam3_deps: dict[str, Any]) -> None:
        """Test predict() raises RuntimeError before fit() is called."""
        model = _build_sam3(mock_sam3_deps, Sam3PromptMode.VISUAL_EXEMPLAR)

        target = Sample(image=torch.zeros(3, 224, 224))

        with pytest.raises(RuntimeError, match="fit"):
            model.predict(target)

    def test_predict_multiple_targets(self, mock_sam3_deps: dict[str, Any]) -> None:
        """Test predict() with multiple target images in visual mode."""
        model = _build_sam3(mock_sam3_deps, Sam3PromptMode.VISUAL_EXEMPLAR)

        ref = Sample(
            image=torch.zeros(3, 224, 224),
            bboxes=np.array([[10, 10, 50, 50]]),
            categories=["shoe"],
            category_ids=np.array([0]),
        )
        model.fit(ref)

        targets = [
            Sample(image=torch.zeros(3, 224, 224)),
            Sample(image=torch.zeros(3, 320, 320)),
        ]
        predictions = model.predict(targets)

        assert len(predictions) == 2


# ---------------------------------------------------------------------------
# Tests: Static utility methods
# ---------------------------------------------------------------------------


class TestSAM3Utilities:
    """Test SAM3 static utility methods."""

    def test_build_category_mapping(self) -> None:
        """Test _build_category_mapping builds correct mapping."""
        samples = [
            Sample(categories=["shoe", "bag"], category_ids=[0, 1]),
            Sample(categories=["hat"], category_ids=[2]),
        ]
        batch = Batch.collate(samples)

        mapping = SAM3._build_category_mapping(batch)

        assert mapping == {"shoe": 0, "bag": 1, "hat": 2}

    def test_build_category_mapping_no_duplicates(self) -> None:
        """Test _build_category_mapping keeps first occurrence."""
        samples = [
            Sample(categories=["shoe"], category_ids=[0]),
            Sample(categories=["shoe"], category_ids=[5]),
        ]
        batch = Batch.collate(samples)

        mapping = SAM3._build_category_mapping(batch)

        assert mapping == {"shoe": 0}

    def test_aggregate_results_with_detections(self) -> None:
        """Test _aggregate_results concatenates non-empty results."""
        masks = [torch.ones(2, 64, 64), torch.ones(1, 64, 64)]
        boxes = [torch.rand(2, 5), torch.rand(1, 5)]
        labels = [torch.tensor([0, 0]), torch.tensor([1])]

        result = SAM3._aggregate_results(masks, boxes, labels, (64, 64))

        assert result["pred_masks"].shape == (3, 64, 64)
        assert result["pred_boxes"].shape == (3, 5)
        assert result["pred_labels"].shape == (3,)

    def test_aggregate_results_empty(self) -> None:
        """Test _aggregate_results with no detections."""
        masks: list[torch.Tensor] = [torch.empty(0, 64, 64)]
        boxes: list[torch.Tensor] = [torch.empty(0, 5)]
        labels: list[torch.Tensor] = [torch.empty(0, dtype=torch.long)]

        result = SAM3._aggregate_results(masks, boxes, labels, (64, 64))

        assert result["pred_masks"].shape == (0, 64, 64)
        assert result["pred_boxes"].shape == (0, 5)
        assert result["pred_labels"].shape == (0,)


# ---------------------------------------------------------------------------
# Tests: Sam3PromptMode enum
# ---------------------------------------------------------------------------


class TestSam3PromptMode:
    """Test Sam3PromptMode enum."""

    def test_classic_value(self) -> None:
        """Test classic mode value."""
        assert Sam3PromptMode.CLASSIC.value == "classic"

    def test_visual_exemplar_value(self) -> None:
        """Test visual exemplar mode value."""
        assert Sam3PromptMode.VISUAL_EXEMPLAR.value == "visual_exemplar"

    def test_construct_from_string(self) -> None:
        """Test enum can be constructed from string values."""
        assert Sam3PromptMode("classic") == Sam3PromptMode.CLASSIC
        assert Sam3PromptMode("visual_exemplar") == Sam3PromptMode.VISUAL_EXEMPLAR

    def test_invalid_string_raises(self) -> None:
        """Test invalid string raises ValueError."""
        with pytest.raises(ValueError):
            Sam3PromptMode("invalid_mode")
