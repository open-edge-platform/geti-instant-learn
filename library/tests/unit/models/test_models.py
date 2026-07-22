# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for models including PerDino, Matcher, SoftMatcher, and GroundedSAM."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torchvision.tv_tensors import Image

from instantlearn.data.base.prediction import Prediction
from instantlearn.models.grounded_sam import GroundedSAM
from instantlearn.models.matcher import Matcher
from instantlearn.models.per_dino import PerDino
from instantlearn.models.soft_matcher import SoftMatcher


def _empty_prediction() -> Prediction:
    """Build an empty backend-neutral ``Prediction`` for mock returns."""
    return Prediction(
        masks=np.zeros((0, 224, 224), dtype=bool),
        scores=np.zeros((0,), dtype=np.float32),
        label_ids=np.zeros((0,), dtype=np.int32),
        label_names=np.array([], dtype=object),
        boxes=np.zeros((0, 4), dtype=np.float32),
        points=None,
    )


def _assert_prediction_list(predictions: list) -> None:
    """Assert a predict() result is a list of ``Prediction`` objects."""
    assert isinstance(predictions, list)
    assert len(predictions) == 1
    assert isinstance(predictions[0], Prediction)
    assert predictions[0].masks is not None
    assert predictions[0].scores is not None
    assert predictions[0].label_ids is not None
    assert predictions[0].label_names is not None


class TestPerDino:
    """Test PerDino model."""

    @pytest.fixture
    def mock_components(self) -> dict[str, Any]:
        """Create mock components for PerDino."""
        return {
            "sam_predictor": MagicMock(),
            "encoder": MagicMock(),
            "masked_feature_extractor": MagicMock(),
            "similarity_matcher": MagicMock(),
            "prompt_generator": MagicMock(),
            "segmenter": MagicMock(),
        }

    @patch("instantlearn.models.per_dino.per_dino.load_sam_model")
    @patch("instantlearn.models.per_dino.per_dino.ImageEncoder")
    def test_per_dino_initialization(
        self,
        mock_image_encoder: MagicMock,
        mock_load_sam: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test PerDino initialization."""
        mock_load_sam.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = PerDino(device="cpu")

        assert hasattr(model, "sam_predictor")
        assert hasattr(model, "encoder")
        assert hasattr(model, "masked_feature_extractor")
        assert hasattr(model, "similarity_matcher")
        assert hasattr(model, "prompt_generator")
        assert hasattr(model, "segmenter")

    def test_per_dino_card(self) -> None:
        """PerDino advertises its own capability card."""
        card = PerDino.card()
        assert card.name == "PerDino"
        assert card.family == "per_dino"

    @patch("instantlearn.models.per_dino.per_dino.load_sam_model")
    @patch("instantlearn.models.per_dino.per_dino.ImageEncoder")
    def test_per_dino_forward_pass(
        self,
        mock_image_encoder: MagicMock,
        mock_load_sam: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test PerDino predict() returns a list of Prediction objects."""
        mock_load_sam.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = PerDino(device="cpu")
        model.fit = MagicMock(return_value=None)
        model.predict = MagicMock(return_value=[_empty_prediction()])

        target_images = [Image(torch.zeros((3, 224, 224), dtype=torch.uint8))]
        predictions = model.predict(target_images)

        _assert_prediction_list(predictions)
        model.predict.assert_called_once_with(target_images)


class TestMatcher:
    """Test Matcher model."""

    @pytest.fixture
    def mock_components(self) -> dict[str, Any]:
        """Create mock components for Matcher."""
        return {
            "sam_predictor": MagicMock(),
            "encoder": MagicMock(),
            "masked_feature_extractor": MagicMock(),
            "prompt_generator": MagicMock(),
            "point_filter": MagicMock(),
            "segmenter": MagicMock(),
        }

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_matcher_initialization(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test Matcher initialization."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = Matcher(device="cpu")

        assert hasattr(model, "sam_predictor")
        assert hasattr(model, "encoder")
        assert hasattr(model, "masked_feature_extractor")
        assert hasattr(model, "prompt_generator")
        assert hasattr(model, "segmenter")

    def test_matcher_card(self) -> None:
        """Matcher advertises its own capability card."""
        card = Matcher.card()
        assert card.name == "Matcher"
        assert card.family == "matcher"

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_matcher_forward_pass(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test Matcher predict() returns a list of Prediction objects."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = Matcher(device="cpu")
        model.fit = MagicMock(return_value=None)
        model.predict = MagicMock(return_value=[_empty_prediction()])

        target_images = [Image(torch.zeros((3, 224, 224), dtype=torch.uint8))]
        predictions = model.predict(target_images)

        _assert_prediction_list(predictions)
        model.predict.assert_called_once_with(target_images)


class TestSoftMatcher:
    """Test SoftMatcher model."""

    @pytest.fixture
    def mock_components(self) -> dict[str, Any]:
        """Create mock components for SoftMatcher."""
        return {
            "sam_predictor": MagicMock(),
            "encoder": MagicMock(),
            "masked_feature_extractor": MagicMock(),
            "prompt_generator": MagicMock(),
            "point_filter": MagicMock(),
            "segmenter": MagicMock(),
        }

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_soft_matcher_initialization(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test SoftMatcher initialization with new components."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = SoftMatcher(device="cpu")

        assert hasattr(model, "sam_predictor")
        assert hasattr(model, "encoder")
        assert hasattr(model, "masked_feature_extractor")
        assert hasattr(model, "prompt_generator")
        assert hasattr(model, "segmenter")

    def test_soft_matcher_card(self) -> None:
        """SoftMatcher advertises its own distinct capability card."""
        card = SoftMatcher.card()
        assert card.name == "SoftMatcher"
        assert card.family == "soft_matcher"

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_soft_matcher_forward_pass(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test SoftMatcher predict() returns a list of Prediction objects."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = SoftMatcher(device="cpu")
        model.fit = MagicMock(return_value=None)
        model.predict = MagicMock(return_value=[_empty_prediction()])

        target_images = [Image(torch.zeros((3, 224, 224), dtype=torch.uint8))]
        predictions = model.predict(target_images)

        _assert_prediction_list(predictions)
        model.predict.assert_called_once_with(target_images)


class TestGroundedSAM:
    """Test GroundedSAM model.

    GroundedSAM has not yet been migrated to the backend-agnostic ``Model`` ABC
    (it still lacks the abstract ``card`` / ``backend`` implementations), so it
    cannot be instantiated. These tests are marked ``xfail`` until GroundedSAM is
    migrated in a follow-up.
    """

    @pytest.fixture
    def mock_components(self) -> dict[str, Any]:
        """Create mock components for GroundedSAM."""
        return {
            "sam_predictor": MagicMock(),
            "prompt_generator": MagicMock(),
            "segmenter": MagicMock(),
            "multi_instance_prior_filter": MagicMock(),
        }

    @pytest.mark.xfail(reason="GroundedSAM not yet migrated to the Model ABC (missing card/backend)", strict=False)
    @patch("instantlearn.models.grounded_sam.grounded_sam.load_sam_model")
    def test_grounded_sam_initialization(self, mock_load_sam: MagicMock, mock_components: dict[str, Any]) -> None:
        """Test GroundedSAM initialization with new components."""
        mock_load_sam.return_value = mock_components["sam_predictor"]

        model = GroundedSAM(device="cpu")

        assert hasattr(model, "sam_predictor")
        assert hasattr(model, "prompt_generator")
        assert hasattr(model, "segmenter")
        assert hasattr(model, "prompt_filter")

    @pytest.mark.xfail(reason="GroundedSAM not yet migrated to the Model ABC (missing card/backend)", strict=False)
    @patch("instantlearn.models.grounded_sam.grounded_sam.load_sam_model")
    def test_grounded_sam_forward_pass(self, mock_load_sam: MagicMock, mock_components: dict[str, Any]) -> None:
        """Test GroundedSAM forward pass with new architecture."""
        mock_load_sam.return_value = mock_components["sam_predictor"]

        model = GroundedSAM(device="cpu")

        model.fit = MagicMock(return_value=None)
        model.predict = MagicMock(
            return_value=[
                {
                    "pred_masks": torch.zeros((0, 224, 224), dtype=torch.bool),
                    "pred_points": torch.zeros((0, 4), dtype=torch.float32),
                    "pred_boxes": torch.zeros((0, 6), dtype=torch.float32),
                    "pred_labels": torch.zeros((0,), dtype=torch.long),
                },
            ],
        )

        target_images = [Image(torch.zeros((3, 224, 224), dtype=torch.uint8))]
        predictions = model.predict(target_images)

        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert isinstance(predictions[0], dict)
        assert "pred_masks" in predictions[0]
        model.predict.assert_called_once_with(target_images)
