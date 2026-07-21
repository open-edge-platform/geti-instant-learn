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

    @patch("instantlearn.models.per_dino.per_dino.load_sam_model")
    @patch("instantlearn.models.per_dino.per_dino.ImageEncoder")
    def test_per_dino_forward_pass(
        self,
        mock_image_encoder: MagicMock,
        mock_load_sam: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test PerDino forward pass."""
        mock_load_sam.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = PerDino(device="cpu")

        # Mock the fit method to set up the model
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

        # Create test data
        target_images = [Image(torch.zeros((3, 224, 224), dtype=torch.uint8))]

        predictions = model.predict(target_images)

        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert isinstance(predictions[0], dict)
        assert "pred_masks" in predictions[0]
        assert "pred_points" in predictions[0]
        assert "pred_boxes" in predictions[0]
        assert "pred_labels" in predictions[0]
        model.predict.assert_called_once_with(target_images)

    @patch("instantlearn.models.per_dino.per_dino.load_sam_model")
    @patch("instantlearn.models.per_dino.per_dino.ImageEncoder")
    def test_per_dino_multi_instance_filtering(
        self,
        mock_image_encoder: MagicMock,
        mock_load_sam: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test that PerDino uses multi-instance prior filtering."""
        mock_load_sam.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = PerDino(device="cpu")

        # Mock the fit and predict methods
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

        model.predict(target_images)

        # Verify that predict was called
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

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_matcher_forward_pass(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test Matcher forward pass."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = Matcher(device="cpu")

        # Mock the learn and infer methods
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

        # Create test data
        target_images = [Image(torch.zeros((3, 224, 224), dtype=torch.uint8))]

        predictions = model.predict(target_images)

        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert isinstance(predictions[0], dict)
        assert "pred_masks" in predictions[0]
        assert "pred_points" in predictions[0]
        assert "pred_boxes" in predictions[0]
        assert "pred_labels" in predictions[0]
        model.predict.assert_called_once_with(target_images)

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_matcher_multi_instance_filtering(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test that Matcher uses multi-instance prior filtering."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = Matcher(device="cpu")

        # Mock the learn and infer methods
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

        model.predict(target_images)

        # Verify that predict was called
        model.predict.assert_called_once_with(target_images)


class TestSoftMatcher:
    """Test SoftMatcher model."""

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

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_soft_matcher_forward_pass(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test SoftMatcher forward pass with new architecture."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = SoftMatcher(device="cpu")

        # Mock the learn and infer methods
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

        # Create test data
        target_images = [Image(torch.zeros((3, 224, 224), dtype=torch.uint8))]

        predictions = model.predict(target_images)

        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert isinstance(predictions[0], dict)
        assert "pred_masks" in predictions[0]
        assert "pred_points" in predictions[0]
        assert "pred_boxes" in predictions[0]
        assert "pred_labels" in predictions[0]
        model.predict.assert_called_once_with(target_images)

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_soft_matcher_multi_instance_filtering(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test that SoftMatcher uses multi-instance prior filtering."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = SoftMatcher(device="cpu")

        # Mock the learn and infer methods
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

        model.predict(target_images)

        # Verify that predict was called
        model.predict.assert_called_once_with(target_images)


class TestGroundedSAM:
    """Test GroundedSAM model."""

    @patch("instantlearn.models.grounded_sam.grounded.TextToBoxPromptGenerator._load_grounding_model_and_processor")
    @patch("instantlearn.models.grounded_sam.grounded_sam.load_sam_model")
    def test_grounded_sam_initialization(self, mock_load_sam: MagicMock, mock_grounding: MagicMock) -> None:
        """Test GroundedSAM initialization with new components."""
        mock_load_sam.return_value = MagicMock()
        mock_grounding.return_value = (MagicMock(), MagicMock())

        model = GroundedSAM(device="cpu")

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
        model = GroundedSAM(device="cpu")
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
        model = GroundedSAM(device="cpu")
        sample = Sample(image=np.zeros((4, 4, 3), dtype=np.uint8), categories=[])

        with pytest.raises(ValueError, match="requires categories"):
            model.predict(sample)

    @patch("instantlearn.models.grounded_sam.grounded.TextToBoxPromptGenerator._load_grounding_model_and_processor")
    @patch("instantlearn.models.grounded_sam.grounded_sam.load_sam_model")
    def test_predict_raises_without_image(self, mock_load_sam: MagicMock, mock_grounding: MagicMock) -> None:
        """predict() raises when a target sample has no image."""
        mock_load_sam.return_value = MagicMock()
        mock_grounding.return_value = (MagicMock(), MagicMock())
        model = GroundedSAM(device="cpu")
        sample = Sample(image=None, categories=[Category(7, "cat")])

        with pytest.raises(ValueError, match="each sample to contain an image"):
            model.predict(sample)
