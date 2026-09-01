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
from instantlearn.data.base.sample import Category, Sample
from instantlearn.models.grounded_sam import GroundedSAM
from instantlearn.models.matcher import Matcher
from instantlearn.models.per_dino import PerDino
from instantlearn.models.soft_matcher import SoftMatcher
from tests import CPU_DEVICE


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

        model = PerDino(device=CPU_DEVICE)

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

        model = PerDino(device=CPU_DEVICE)
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

        model = Matcher(device=CPU_DEVICE)

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

        model = Matcher(device=CPU_DEVICE)
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

        model = SoftMatcher(device=CPU_DEVICE)

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

        model = SoftMatcher(device=CPU_DEVICE)
        model.fit = MagicMock(return_value=None)
        model.predict = MagicMock(return_value=[_empty_prediction()])

        target_images = [Image(torch.zeros((3, 224, 224), dtype=torch.uint8))]
        predictions = model.predict(target_images)

        _assert_prediction_list(predictions)
        model.predict.assert_called_once_with(target_images)


class TestGroundedSAM:
    """Test GroundedSAM model."""

    @patch("instantlearn.models.grounded_sam.grounded.TextToBoxPromptGenerator._load_grounding_model_and_processor")
    @patch("instantlearn.models.grounded_sam.grounded_sam.load_sam_model")
    def test_grounded_sam_initialization(self, mock_load_sam: MagicMock, mock_grounding: MagicMock) -> None:
        """Test GroundedSAM initialization with new components."""
        mock_load_sam.return_value = MagicMock()
        mock_grounding.return_value = (MagicMock(), MagicMock())

        model = GroundedSAM(device=CPU_DEVICE)

        assert hasattr(model, "sam_predictor")
        assert hasattr(model, "prompt_generator")
        assert hasattr(model, "segmenter")
        assert hasattr(model, "prompt_filter")

    @patch("instantlearn.models.grounded_sam.grounded.TextToBoxPromptGenerator._load_grounding_model_and_processor")
    @patch("instantlearn.models.grounded_sam.grounded_sam.load_sam_model")
    def test_predict_returns_predictions(self, mock_load_sam: MagicMock, mock_grounding: MagicMock) -> None:
        """predict() converts segmenter dicts to Prediction using target categories."""
        mock_load_sam.return_value = MagicMock()
        mock_grounding.return_value = (MagicMock(), MagicMock())
        model = GroundedSAM(device=CPU_DEVICE)
        model.postprocessor = None
        model.prompt_generator = MagicMock(
            spec=torch.nn.Module,
            return_value=(torch.zeros(1, 1, 1, 5), torch.tensor([7])),
        )
        model.prompt_filter = MagicMock(spec=torch.nn.Module, side_effect=lambda box_prompts: box_prompts)
        model.segmenter = MagicMock(
            spec=torch.nn.Module,
            return_value=[
                {
                    "pred_masks": torch.ones(1, 4, 4),
                    "pred_labels": torch.tensor([7]),
                    "pred_scores": torch.tensor([0.9]),
                },
            ],
        )

        sample = Sample(image=np.zeros((4, 4, 3), dtype=np.uint8), categories=[Category(7, "cat")])
        predictions = model.predict(sample)

        assert len(predictions) == 1
        assert isinstance(predictions[0], Prediction)
        assert predictions[0].label_ids.tolist() == [7]
        assert predictions[0].label_names.tolist() == ["cat"]
        assert predictions[0].masks.shape == (1, 4, 4)

    @patch("instantlearn.models.grounded_sam.grounded.TextToBoxPromptGenerator._load_grounding_model_and_processor")
    @patch("instantlearn.models.grounded_sam.grounded_sam.load_sam_model")
    def test_predict_raises_without_categories(self, mock_load_sam: MagicMock, mock_grounding: MagicMock) -> None:
        """predict() raises when neither fit() nor the target provide categories."""
        mock_load_sam.return_value = MagicMock()
        mock_grounding.return_value = (MagicMock(), MagicMock())
        model = GroundedSAM(device=CPU_DEVICE)
        sample = Sample(image=np.zeros((4, 4, 3), dtype=np.uint8), categories=[])

        with pytest.raises(ValueError, match="requires categories"):
            model.predict(sample)

    @patch("instantlearn.models.grounded_sam.grounded.TextToBoxPromptGenerator._load_grounding_model_and_processor")
    @patch("instantlearn.models.grounded_sam.grounded_sam.load_sam_model")
    def test_predict_raises_without_image(self, mock_load_sam: MagicMock, mock_grounding: MagicMock) -> None:
        """predict() raises when a target sample has no image."""
        mock_load_sam.return_value = MagicMock()
        mock_grounding.return_value = (MagicMock(), MagicMock())
        model = GroundedSAM(device=CPU_DEVICE)
        sample = Sample(image=None, categories=[Category(7, "cat")])

        with pytest.raises(ValueError, match="each sample to contain an image"):
            model.predict(sample)
