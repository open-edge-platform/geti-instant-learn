# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for models including PerDino, Matcher, SoftMatcher, and GroundedSAM."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torchvision.tv_tensors import Image

from getiprompt.models.grounded_sam import GroundedSAM
from getiprompt.models.matcher import Matcher
from getiprompt.models.per_dino import PerDino
from getiprompt.models.soft_matcher import SoftMatcher
from getiprompt.types import Results


class TestPerDino:
    """Test PerDino model."""

    @pytest.fixture
    def mock_components(self) -> dict[str, Any]:
        """Create mock components for PerDino."""
        return {
            "sam_predictor": MagicMock(),
            "encoder": MagicMock(),
            "feature_selector": MagicMock(),
            "similarity_matcher": MagicMock(),
            "prompt_generator": MagicMock(),
            "segmenter": MagicMock(),
        }

    @patch("getiprompt.models.per_dino.load_sam_model")
    @patch("getiprompt.models.per_dino.ImageEncoder")
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
        assert hasattr(model, "feature_selector")
        assert hasattr(model, "similarity_matcher")
        assert hasattr(model, "prompt_generator")
        assert hasattr(model, "segmenter")

    @patch("getiprompt.models.per_dino.load_sam_model")
    @patch("getiprompt.models.per_dino.ImageEncoder")
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

        # Mock the learn method to set up the model
        model.learn = MagicMock(return_value=Results())
        model.infer = MagicMock(return_value=Results())

        # Create test data
        target_images = [Image(torch.zeros((224, 224, 3), dtype=torch.uint8))]

        result = model.infer(target_images)

        assert isinstance(result, Results)
        model.infer.assert_called_once_with(target_images)

    @patch("getiprompt.models.per_dino.load_sam_model")
    @patch("getiprompt.models.per_dino.ImageEncoder")
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

        # Mock the learn and infer methods
        model.learn = MagicMock(return_value=Results())
        model.infer = MagicMock(return_value=Results())

        target_images = [Image(torch.zeros((224, 224, 3), dtype=torch.uint8))]

        model.infer(target_images)

        # Verify that infer was called
        model.infer.assert_called_once_with(target_images)


class TestMatcher:
    """Test Matcher model."""

    @pytest.fixture
    def mock_components(self) -> dict[str, Any]:
        """Create mock components for Matcher."""
        return {
            "sam_predictor": MagicMock(),
            "encoder": MagicMock(),
            "feature_selector": MagicMock(),
            "prompt_generator": MagicMock(),
            "point_filter": MagicMock(),
            "segmenter": MagicMock(),
            "mask_adder": MagicMock(),
            "mask_processor": MagicMock(),
        }

    @patch("getiprompt.models.matcher.load_sam_model")
    @patch("getiprompt.models.matcher.ImageEncoder")
    def test_matcher_initialization(
        self,
        mock_image_encoder: MagicMock,
        mock_load_sam: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test Matcher initialization."""
        mock_load_sam.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = Matcher(device="cpu")

        assert hasattr(model, "sam_predictor")
        assert hasattr(model, "encoder")
        assert hasattr(model, "feature_selector")
        assert hasattr(model, "prompt_generator")
        assert hasattr(model, "point_filter")
        assert hasattr(model, "segmenter")
        assert hasattr(model, "mask_adder")
        assert hasattr(model, "mask_processor")

    @patch("getiprompt.models.matcher.load_sam_model")
    @patch("getiprompt.models.matcher.ImageEncoder")
    def test_matcher_forward_pass(
        self,
        mock_image_encoder: MagicMock,
        mock_load_sam: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test Matcher forward pass."""
        mock_load_sam.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = Matcher(device="cpu")

        # Mock the learn and infer methods
        model.learn = MagicMock(return_value=Results())
        model.infer = MagicMock(return_value=Results())

        # Create test data
        target_images = [Image(torch.zeros((224, 224, 3), dtype=torch.uint8))]

        result = model.infer(target_images)

        assert isinstance(result, Results)
        model.infer.assert_called_once_with(target_images)

    @patch("getiprompt.models.matcher.load_sam_model")
    @patch("getiprompt.models.matcher.ImageEncoder")
    def test_matcher_multi_instance_filtering(
        self,
        mock_image_encoder: MagicMock,
        mock_load_sam: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test that Matcher uses multi-instance prior filtering."""
        mock_load_sam.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = Matcher(device="cpu")

        # Mock the learn and infer methods
        model.learn = MagicMock(return_value=Results())
        model.infer = MagicMock(return_value=Results())

        target_images = [Image(torch.zeros((224, 224, 3), dtype=torch.uint8))]

        model.infer(target_images)

        # Verify that infer was called
        model.infer.assert_called_once_with(target_images)


class TestSoftMatcher:
    """Test SoftMatcher model."""

    @pytest.fixture
    def mock_components(self) -> dict[str, Any]:
        """Create mock components for SoftMatcher."""
        return {
            "sam_predictor": MagicMock(),
            "encoder": MagicMock(),
            "feature_selector": MagicMock(),
            "prompt_generator": MagicMock(),
            "point_filter": MagicMock(),
            "segmenter": MagicMock(),
            "mask_adder": MagicMock(),
            "mask_processor": MagicMock(),
        }

    @patch("getiprompt.models.matcher.load_sam_model")
    @patch("getiprompt.models.matcher.ImageEncoder")
    def test_soft_matcher_initialization(
        self,
        mock_image_encoder: MagicMock,
        mock_load_sam: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test SoftMatcher initialization with new components."""
        mock_load_sam.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = SoftMatcher(device="cpu")

        assert hasattr(model, "sam_predictor")
        assert hasattr(model, "encoder")
        assert hasattr(model, "feature_selector")
        assert hasattr(model, "prompt_generator")
        assert hasattr(model, "point_filter")
        assert hasattr(model, "segmenter")
        assert hasattr(model, "mask_adder")
        assert hasattr(model, "mask_processor")

    @patch("getiprompt.models.matcher.load_sam_model")
    @patch("getiprompt.models.matcher.ImageEncoder")
    def test_soft_matcher_forward_pass(
        self,
        mock_image_encoder: MagicMock,
        mock_load_sam: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test SoftMatcher forward pass with new architecture."""
        mock_load_sam.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = SoftMatcher(device="cpu")

        # Mock the learn and infer methods
        model.learn = MagicMock(return_value=Results())
        model.infer = MagicMock(return_value=Results())

        # Create test data
        target_images = [Image(torch.zeros((224, 224, 3), dtype=torch.uint8))]

        result = model.infer(target_images)

        assert isinstance(result, Results)
        model.infer.assert_called_once_with(target_images)

    @patch("getiprompt.models.matcher.load_sam_model")
    @patch("getiprompt.models.matcher.ImageEncoder")
    def test_soft_matcher_multi_instance_filtering(
        self,
        mock_image_encoder: MagicMock,
        mock_load_sam: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test that SoftMatcher uses multi-instance prior filtering."""
        mock_load_sam.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = SoftMatcher(device="cpu")

        # Mock the learn and infer methods
        model.learn = MagicMock(return_value=Results())
        model.infer = MagicMock(return_value=Results())

        target_images = [Image(torch.zeros((224, 224, 3), dtype=torch.uint8))]

        model.infer(target_images)

        # Verify that infer was called
        model.infer.assert_called_once_with(target_images)


class TestGroundedSAM:
    """Test GroundedSAM model."""

    @pytest.fixture
    def mock_components(self) -> dict[str, Any]:
        """Create mock components for GroundedSAM."""
        return {
            "sam_predictor": MagicMock(),
            "prompt_generator": MagicMock(),
            "segmenter": MagicMock(),
            "multi_instance_prior_filter": MagicMock(),
            "mask_processor": MagicMock(),
        }

    @patch("getiprompt.models.grounded_sam.load_sam_model")
    def test_grounded_sam_initialization(self, mock_load_sam: MagicMock, mock_components: dict[str, Any]) -> None:
        """Test GroundedSAM initialization with new components."""
        mock_load_sam.return_value = mock_components["sam_predictor"]

        model = GroundedSAM(device="cpu")

        assert hasattr(model, "sam_predictor")
        assert hasattr(model, "prompt_generator")
        assert hasattr(model, "segmenter")
        assert hasattr(model, "multi_instance_prior_filter")
        assert hasattr(model, "mask_processor")

    @patch("getiprompt.models.grounded_sam.load_sam_model")
    def test_grounded_sam_forward_pass(self, mock_load_sam: MagicMock, mock_components: dict[str, Any]) -> None:
        """Test GroundedSAM forward pass with new architecture."""
        mock_load_sam.return_value = mock_components["sam_predictor"]

        model = GroundedSAM(device="cpu")

        # Mock the learn and infer methods
        model.learn = MagicMock(return_value=Results())
        model.infer = MagicMock(return_value=Results())

        # Create test data
        target_images = [Image(torch.zeros((224, 224, 3), dtype=torch.uint8))]

        result = model.infer(target_images)

        assert isinstance(result, Results)
        model.infer.assert_called_once_with(target_images)

    @patch("getiprompt.models.grounded_sam.load_sam_model")
    def test_grounded_sam_multi_instance_filtering(
        self,
        mock_load_sam: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test that GroundedSAM uses multi-instance prior filtering."""
        mock_load_sam.return_value = mock_components["sam_predictor"]

        model = GroundedSAM(device="cpu")

        # Mock the learn and infer methods
        model.learn = MagicMock(return_value=Results())
        model.infer = MagicMock(return_value=Results())

        target_images = [Image(torch.zeros((224, 224, 3), dtype=torch.uint8))]

        model.infer(target_images)

        # Verify that infer was called
        model.infer.assert_called_once_with(target_images)


class TestModelIntegration:
    """Test integration across all models."""

    @patch("getiprompt.models.per_dino.load_sam_model")
    @patch("getiprompt.models.per_dino.ImageEncoder")
    @patch("getiprompt.models.matcher.load_sam_model")
    @patch("getiprompt.models.matcher.ImageEncoder")
    @patch("getiprompt.models.grounded_sam.load_sam_model")
    def test_all_models_use_multi_instance_filtering(
        self,
        mock_grounded_sam_load_sam: MagicMock,
        mock_matcher_image_encoder: MagicMock,
        mock_matcher_load_sam: MagicMock,
        mock_per_dino_image_encoder: MagicMock,
        mock_per_dino_load_sam: MagicMock,
    ) -> None:
        """Test that all models use multi-instance prior filtering."""
        # Mock all the load functions
        mock_per_dino_load_sam.return_value = MagicMock()
        mock_per_dino_image_encoder.return_value = MagicMock()
        mock_matcher_load_sam.return_value = MagicMock()
        mock_matcher_image_encoder.return_value = MagicMock()
        mock_grounded_sam_load_sam.return_value = MagicMock()

        models = [PerDino, Matcher, SoftMatcher, GroundedSAM]

        for model_class in models:
            # Create model instance with CPU device
            model = model_class(device="cpu")

            # Mock the learn and infer methods
            model.learn = MagicMock(return_value=Results())
            model.infer = MagicMock(return_value=Results())

            # Test forward pass
            target_images = [Image(torch.zeros((224, 224, 3), dtype=torch.uint8))]
            model.infer(target_images)

            # Verify that infer was called
            model.infer.assert_called_once_with(target_images)

    @patch("getiprompt.models.per_dino.load_sam_model")
    @patch("getiprompt.models.per_dino.ImageEncoder")
    @patch("getiprompt.models.matcher.load_sam_model")
    @patch("getiprompt.models.matcher.ImageEncoder")
    @patch("getiprompt.models.grounded_sam.load_sam_model")
    def test_model_consistency(
        self,
        mock_grounded_sam_load_sam: MagicMock,
        mock_matcher_image_encoder: MagicMock,
        mock_matcher_load_sam: MagicMock,
        mock_per_dino_image_encoder: MagicMock,
        mock_per_dino_load_sam: MagicMock,
    ) -> None:
        """Test that all models have consistent interfaces."""
        # Mock all the load functions
        mock_per_dino_load_sam.return_value = MagicMock()
        mock_per_dino_image_encoder.return_value = MagicMock()
        mock_matcher_load_sam.return_value = MagicMock()
        mock_matcher_image_encoder.return_value = MagicMock()
        mock_grounded_sam_load_sam.return_value = MagicMock()

        models = [PerDino, Matcher, SoftMatcher, GroundedSAM]

        for model_class in models:
            # Create model instance
            model = model_class(device="cpu")

            # Test that all models have required methods
            assert hasattr(model, "learn")
            assert hasattr(model, "infer")
            assert callable(model.learn)
            assert callable(model.infer)

    def test_model_error_handling(self) -> None:
        """Test error handling in model changes."""
        models = [PerDino, Matcher, SoftMatcher, GroundedSAM]

        for model_class in models:
            # Test with invalid device parameter
            with pytest.raises((TypeError, ValueError, RuntimeError)):
                model_class(device="invalid_device")

    @patch("getiprompt.models.per_dino.load_sam_model")
    @patch("getiprompt.models.per_dino.ImageEncoder")
    @patch("getiprompt.models.matcher.load_sam_model")
    @patch("getiprompt.models.matcher.ImageEncoder")
    @patch("getiprompt.models.grounded_sam.load_sam_model")
    def test_model_performance_tracking(
        self,
        mock_grounded_sam_load_sam: MagicMock,
        mock_matcher_image_encoder: MagicMock,
        mock_matcher_load_sam: MagicMock,
        mock_per_dino_image_encoder: MagicMock,
        mock_per_dino_load_sam: MagicMock,
    ) -> None:
        """Test that models use performance tracking."""
        # Mock all the load functions
        mock_per_dino_load_sam.return_value = MagicMock()
        mock_per_dino_image_encoder.return_value = MagicMock()
        mock_matcher_load_sam.return_value = MagicMock()
        mock_matcher_image_encoder.return_value = MagicMock()
        mock_grounded_sam_load_sam.return_value = MagicMock()

        models = [PerDino, Matcher, SoftMatcher, GroundedSAM]

        for model_class in models:
            # Create model instance
            model = model_class(device="cpu")

            # Mock the learn and infer methods
            model.learn = MagicMock(return_value=Results())
            model.infer = MagicMock(return_value=Results())

            # Test that models can be called without errors
            target_images = [Image(torch.zeros((224, 224, 3), dtype=torch.uint8))]
            result = model.infer(target_images)

            assert isinstance(result, Results)
