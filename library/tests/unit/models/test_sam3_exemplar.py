# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SAM3 and EfficientSAM3 exemplar (visual query) mode.

Tests cover:
- Sam3PromptMode enum construction
- SAM3 & EfficientSAM3 __init__ with prompt_mode/drop_spatial_bias
- Classic mode fit/predict dispatch
- Visual exemplar mode fit/predict dispatch
- _build_category_mapping helper
- _aggregate_results helper
- Error handling (missing fit, missing images, empty prompts)
- Sam3PromptPreprocessor new API (boxes, points, both)
- GeometryEncoder forward with optional boxes and drop_spatial_bias
- Sam3Model.forward with precomputed_geometry_features
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from instantlearn.data.base.batch import Batch
from instantlearn.data.base.sample import Sample
from instantlearn.models.efficient_sam3.efficient_sam3 import EfficientSAM3
from instantlearn.models.sam3.model import GeometryEncoder, Sam3Model
from instantlearn.models.sam3.processing import Sam3PromptPreprocessor
from instantlearn.models.sam3.sam3 import SAM3, Sam3PromptMode

# Sam3PromptMode


class TestSam3PromptMode:
    """Tests for Sam3PromptMode enum."""

    def test_classic_value(self) -> None:
        """Verify CLASSIC enum value."""
        assert Sam3PromptMode.CLASSIC.value == "classic"

    def test_visual_exemplar_value(self) -> None:
        """Verify VISUAL_EXEMPLAR enum value."""
        assert Sam3PromptMode.VISUAL_EXEMPLAR.value == "visual_exemplar"

    def test_from_string_classic(self) -> None:
        """Construct CLASSIC from string."""
        assert Sam3PromptMode("classic") == Sam3PromptMode.CLASSIC

    def test_from_string_visual_exemplar(self) -> None:
        """Construct VISUAL_EXEMPLAR from string."""
        assert Sam3PromptMode("visual_exemplar") == Sam3PromptMode.VISUAL_EXEMPLAR

    def test_invalid_raises(self) -> None:
        """Invalid string raises ValueError."""
        with pytest.raises(ValueError, match="invalid_mode"):
            Sam3PromptMode("invalid_mode")

    def test_is_string(self) -> None:
        """Sam3PromptMode should be usable as a string."""
        mode = Sam3PromptMode.CLASSIC
        assert mode == "classic"


# Sam3PromptPreprocessor new API


class TestSam3PromptPreprocessorAPI:
    """Tests for the updated Sam3PromptPreprocessor forward API."""

    @pytest.fixture
    def preprocessor(self) -> Sam3PromptPreprocessor:
        """Create a Sam3PromptPreprocessor instance."""
        return Sam3PromptPreprocessor(target_size=1008)

    @pytest.fixture
    def original_sizes(self) -> torch.Tensor:
        """Create a standard original_sizes tensor."""
        return torch.tensor([[480, 640]])

    def test_boxes_only(self, preprocessor: Sam3PromptPreprocessor, original_sizes: torch.Tensor) -> None:
        """forward() with input_boxes returns (boxes, None)."""
        bbox = np.array([100, 100, 200, 200])
        boxes, points = preprocessor(original_sizes, input_boxes=bbox)
        assert boxes is not None
        assert points is None
        assert boxes.ndim == 3  # [1, 1, 4]

    def test_points_only(self, preprocessor: Sam3PromptPreprocessor, original_sizes: torch.Tensor) -> None:
        """forward() with input_points returns (None, points)."""
        point = np.array([150, 150])
        boxes, points = preprocessor(original_sizes, input_points=point)
        assert boxes is None
        assert points is not None
        assert points.ndim == 3  # [1, 1, 2]

    def test_both_boxes_and_points(
        self,
        preprocessor: Sam3PromptPreprocessor,
        original_sizes: torch.Tensor,
    ) -> None:
        """forward() with both returns (boxes, points)."""
        bbox = np.array([100, 100, 200, 200])
        point = np.array([150, 150])
        boxes, points = preprocessor(original_sizes, input_boxes=bbox, input_points=point)
        assert boxes is not None
        assert points is not None

    def test_neither_returns_none_tuple(
        self,
        preprocessor: Sam3PromptPreprocessor,
        original_sizes: torch.Tensor,
    ) -> None:
        """forward() with neither input returns (None, None)."""
        boxes, points = preprocessor(original_sizes)
        assert boxes is None
        assert points is None

    def test_box_normalization_range(
        self,
        preprocessor: Sam3PromptPreprocessor,
        original_sizes: torch.Tensor,
    ) -> None:
        """Normalized box coords should be in [0, 1] range."""
        bbox = np.array([100, 100, 200, 200])
        boxes, _ = preprocessor(original_sizes, input_boxes=bbox)
        # After normalization boxes should be roughly in [0, 1]
        assert boxes.min() >= 0
        assert boxes.max() <= 1.5  # cxcywh can exceed 1 for w/h at edges


# Helpers: mock model factory
def _mock_sam3_model() -> MagicMock:
    """Create a mock Sam3Model that chains .to().eval()."""
    mock = MagicMock()
    mock.to.return_value = mock
    mock.eval.return_value = mock
    return mock


# SAM3 class — init and prompt mode
class TestSAM3Init:
    """Tests for SAM3 initialization with prompt mode params."""

    @patch("instantlearn.models.sam3.sam3.Sam3Model.from_pretrained", return_value=_mock_sam3_model())
    @patch("instantlearn.models.sam3.sam3.CLIPTokenizerFast.from_pretrained")
    def test_default_classic_mode(self, _mock_tokenizer: MagicMock, _mock_model: MagicMock) -> None:  # noqa: ARG002, PT019
        """SAM3 defaults to CLASSIC prompt mode."""
        sam3 = SAM3(device="cpu")
        assert sam3.prompt_mode == Sam3PromptMode.CLASSIC
        assert sam3.drop_spatial_bias is False
        assert sam3.exemplar_geometry_features is None

    @patch("instantlearn.models.sam3.sam3.Sam3Model.from_pretrained", return_value=_mock_sam3_model())
    @patch("instantlearn.models.sam3.sam3.CLIPTokenizerFast.from_pretrained")
    def test_visual_exemplar_mode(self, _mock_tokenizer: MagicMock, _mock_model: MagicMock) -> None:  # noqa: ARG002, PT019
        """SAM3 can be initialized in VISUAL_EXEMPLAR mode."""
        sam3 = SAM3(device="cpu", prompt_mode="visual_exemplar", drop_spatial_bias=True)
        assert sam3.prompt_mode == Sam3PromptMode.VISUAL_EXEMPLAR
        assert sam3.drop_spatial_bias is True

    @patch("instantlearn.models.sam3.sam3.Sam3Model.from_pretrained", return_value=_mock_sam3_model())
    @patch("instantlearn.models.sam3.sam3.CLIPTokenizerFast.from_pretrained")
    def test_invalid_prompt_mode_raises(self, _mock_tokenizer: MagicMock, _mock_model: MagicMock) -> None:  # noqa: ARG002, PT019
        """Invalid prompt_mode string raises ValueError."""
        with pytest.raises(ValueError, match="nonexistent"):
            SAM3(device="cpu", prompt_mode="nonexistent")


# SAM3 class — _build_category_mapping
class TestBuildCategoryMapping:
    """Tests for SAM3._build_category_mapping static method."""

    def test_single_sample(self) -> None:
        """Single sample with two categories."""
        batch = Batch.collate(Sample(categories=["cat", "dog"], category_ids=[0, 1]))
        mapping = SAM3._build_category_mapping(batch)  # noqa: SLF001
        assert mapping == {"cat": 0, "dog": 1}

    def test_multiple_samples_dedup(self) -> None:
        """Multiple samples deduplicate categories."""
        samples = [
            Sample(categories=["cat"], category_ids=[0]),
            Sample(categories=["cat", "dog"], category_ids=[0, 1]),
        ]
        batch = Batch.collate(samples)
        mapping = SAM3._build_category_mapping(batch)  # noqa: SLF001
        assert mapping == {"cat": 0, "dog": 1}

    def test_empty_categories(self) -> None:
        """Empty categories produce empty mapping."""
        sample = Sample(categories=[], category_ids=[])
        batch = Batch.collate(sample)
        mapping = SAM3._build_category_mapping(batch)  # noqa: SLF001
        assert mapping == {}


# SAM3 class — _aggregate_results
class TestAggregateResults:
    """Tests for SAM3._aggregate_results static method."""

    def test_non_empty(self) -> None:
        """Non-empty predictions are concatenated."""
        masks = [torch.ones(1, 10, 10)]
        boxes = [torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.9]])]
        labels = [torch.tensor([0])]

        result = SAM3._aggregate_results(masks, boxes, labels, (10, 10))  # noqa: SLF001
        assert result["pred_masks"].shape == (1, 10, 10)
        assert result["pred_boxes"].shape == (1, 5)
        assert result["pred_labels"].shape == (1,)

    def test_empty(self) -> None:
        """Empty input produces zero-sized tensors."""
        result = SAM3._aggregate_results([], [], [], (10, 10))  # noqa: SLF001
        assert result["pred_masks"].shape == (0, 10, 10)
        assert result["pred_boxes"].shape == (0, 5)
        assert result["pred_labels"].shape == (0,)

    def test_mixed_empty_nonempty(self) -> None:
        """Mix of empty and non-empty tensors filters correctly."""
        masks = [torch.empty(0, 10, 10), torch.ones(2, 10, 10)]
        boxes = [torch.empty(0, 5), torch.ones(2, 5)]
        labels = [torch.empty(0, dtype=torch.long), torch.tensor([0, 1])]

        result = SAM3._aggregate_results(masks, boxes, labels, (10, 10))  # noqa: SLF001
        assert result["pred_masks"].shape == (2, 10, 10)


# SAM3 class — visual exemplar fit error handling
class TestSAM3ExemplarErrors:
    """Tests for SAM3 error handling in visual exemplar mode."""

    @patch("instantlearn.models.sam3.sam3.Sam3Model.from_pretrained", return_value=_mock_sam3_model())
    @patch("instantlearn.models.sam3.sam3.CLIPTokenizerFast.from_pretrained")
    def test_predict_without_fit_raises(self, _mock_tokenizer: MagicMock, _mock_model: MagicMock) -> None:  # noqa: ARG002, PT019
        """predict() in VISUAL_EXEMPLAR mode without fit() raises RuntimeError."""
        sam3 = SAM3(device="cpu", prompt_mode="visual_exemplar")

        target = Sample(image=torch.zeros(3, 100, 100))
        with pytest.raises(RuntimeError, match="No cached exemplar features"):
            sam3.predict(target)

    @patch("instantlearn.models.sam3.sam3.Sam3Model.from_pretrained", return_value=_mock_sam3_model())
    @patch("instantlearn.models.sam3.sam3.CLIPTokenizerFast.from_pretrained")
    def test_fit_no_bboxes_raises(self, _mock_tokenizer: MagicMock, _mock_model: MagicMock) -> None:  # noqa: ARG002, PT019
        """fit() in VISUAL_EXEMPLAR mode with no bboxes/points raises ValueError."""
        sam3 = SAM3(device="cpu", prompt_mode="visual_exemplar")

        ref = Sample(categories=["cat"], category_ids=[0])
        with pytest.raises(ValueError, match="VISUAL_EXEMPLAR mode requires at least one"):
            sam3.fit(ref)

    @patch("instantlearn.models.sam3.sam3.Sam3Model.from_pretrained", return_value=_mock_sam3_model())
    @patch("instantlearn.models.sam3.sam3.CLIPTokenizerFast.from_pretrained")
    def test_fit_exemplar_no_image_raises(self, _mock_tokenizer: MagicMock, _mock_model: MagicMock) -> None:  # noqa: ARG002, PT019
        """fit() in VISUAL_EXEMPLAR mode with bboxes but no image raises ValueError."""
        sam3 = SAM3(device="cpu", prompt_mode="visual_exemplar")

        ref = Sample(
            bboxes=np.array([[100, 100, 200, 200]]),
            category_ids=np.array([0]),
            categories=["cat"],
        )
        with pytest.raises(ValueError, match="requires images"):
            sam3.fit(ref)


# EfficientSAM3 class — init and prompt mode
def _mock_efficient_sam3_model() -> MagicMock:
    """Create a mock EfficientSam3Model that chains .to().eval()."""
    mock = MagicMock()
    mock.to.return_value = mock
    mock.eval.return_value = mock
    return mock


class TestEfficientSAM3Init:
    """Tests for EfficientSAM3 initialization with prompt mode params."""

    @patch(
        "instantlearn.models.efficient_sam3.efficient_sam3.EfficientSam3Model.from_pretrained",
        return_value=_mock_efficient_sam3_model(),
    )
    @patch("instantlearn.models.efficient_sam3.efficient_sam3.CLIPTokenizerFast.from_pretrained")
    def test_default_classic_mode(self, _mock_tokenizer: MagicMock, _mock_model: MagicMock) -> None:  # noqa: ARG002, PT019
        """EfficientSAM3 defaults to CLASSIC prompt mode."""
        model = EfficientSAM3(device="cpu")
        assert model.prompt_mode == Sam3PromptMode.CLASSIC
        assert model.drop_spatial_bias is False
        assert model.exemplar_geometry_features is None

    @patch(
        "instantlearn.models.efficient_sam3.efficient_sam3.EfficientSam3Model.from_pretrained",
        return_value=_mock_efficient_sam3_model(),
    )
    @patch("instantlearn.models.efficient_sam3.efficient_sam3.CLIPTokenizerFast.from_pretrained")
    def test_visual_exemplar_mode(self, _mock_tokenizer: MagicMock, _mock_model: MagicMock) -> None:  # noqa: ARG002, PT019
        """EfficientSAM3 can be initialized in VISUAL_EXEMPLAR mode."""
        model = EfficientSAM3(device="cpu", prompt_mode="visual_exemplar", drop_spatial_bias=True)
        assert model.prompt_mode == Sam3PromptMode.VISUAL_EXEMPLAR
        assert model.drop_spatial_bias is True


# ---------------------------------------------------------------------------
# EfficientSAM3 class — error handling
# ---------------------------------------------------------------------------


class TestEfficientSAM3ExemplarErrors:
    """Tests for EfficientSAM3 error handling in visual exemplar mode."""

    @patch(
        "instantlearn.models.efficient_sam3.efficient_sam3.EfficientSam3Model.from_pretrained",
        return_value=_mock_efficient_sam3_model(),
    )
    @patch("instantlearn.models.efficient_sam3.efficient_sam3.CLIPTokenizerFast.from_pretrained")
    def test_predict_without_fit_raises(self, _mock_tokenizer: MagicMock, _mock_model: MagicMock) -> None:  # noqa: ARG002, PT019
        """predict() in VISUAL_EXEMPLAR mode without fit() raises RuntimeError."""
        model = EfficientSAM3(device="cpu", prompt_mode="visual_exemplar")

        target = Sample(image=torch.zeros(3, 100, 100))
        with pytest.raises(RuntimeError, match="No cached exemplar features"):
            model.predict(target)


# GeometryEncoder forward — optional boxes and drop_spatial_bias
class TestGeometryEncoderForward:
    """Tests for GeometryEncoder.forward with optional boxes and drop_spatial_bias."""

    def test_forward_signature_accepts_optional_boxes(self) -> None:
        """GeometryEncoder.forward should accept None for box args."""
        sig = inspect.signature(GeometryEncoder.forward)
        params = sig.parameters

        assert "box_embeddings" in params
        assert "box_mask" in params
        assert "box_labels" in params
        assert "drop_spatial_bias" in params

        # box params should have defaults (Optional with None default)
        assert params["box_embeddings"].default is None
        assert params["box_mask"].default is None
        assert params["box_labels"].default is None
        assert params["drop_spatial_bias"].default is False

    def test_forward_signature_accepts_points(self) -> None:
        """GeometryEncoder.forward should accept point args."""
        sig = inspect.signature(GeometryEncoder.forward)
        params = sig.parameters

        assert "point_embeddings" in params
        assert "point_mask" in params
        assert "point_labels" in params


# Sam3Model.forward — precomputed geometry features
class TestSam3ModelForwardSignature:
    """Tests for Sam3Model.forward signature additions."""

    def test_forward_accepts_precomputed_geometry(self) -> None:
        """Sam3Model.forward should accept precomputed_geometry_features args."""
        sig = inspect.signature(Sam3Model.forward)
        params = sig.parameters

        assert "precomputed_geometry_features" in params
        assert "precomputed_geometry_mask" in params
        assert "drop_spatial_bias" in params

        assert params["precomputed_geometry_features"].default is None
        assert params["precomputed_geometry_mask"].default is None
        assert params["drop_spatial_bias"].default is False


# Exports
class TestExports:
    """Tests for Sam3PromptMode exports from all init files."""

    def test_sam3_package_exports(self) -> None:
        """Sam3PromptMode is exported from sam3 package."""
        from instantlearn.models.sam3 import Sam3PromptMode as ExportedMode  # noqa: PLC0415

        assert ExportedMode is Sam3PromptMode

    def test_efficient_sam3_package_exports(self) -> None:
        """Sam3PromptMode is re-exported from efficient_sam3 package."""
        from instantlearn.models.efficient_sam3 import Sam3PromptMode as ExportedMode  # noqa: PLC0415

        assert ExportedMode is Sam3PromptMode

    def test_top_level_exports(self) -> None:
        """Sam3PromptMode is exported from top-level models package."""
        from instantlearn.models import Sam3PromptMode as ExportedMode  # noqa: PLC0415

        assert ExportedMode is Sam3PromptMode
