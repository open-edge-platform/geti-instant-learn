# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Encoder."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from getiprompt.processes.encoders.encoder import ENCODER_MODEL_COLLECTION, Encoder
from getiprompt.types import Features, Image, Masks, Priors


class TestEncoder:
    """Test the Encoder class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Mock the HuggingFace model loading
        self.mock_model = Mock()
        self.mock_model.config.patch_size = 16
        self.mock_model.config.num_register_tokens = 0
        self.mock_model.device = torch.device("cpu")

        # Mock the processor
        self.mock_processor = Mock()

        # Mock the model's forward method
        self.mock_model.return_value = Mock()
        self.mock_model.return_value.last_hidden_state = torch.randn(1, 197, 1024)  # [batch, seq_len, hidden_dim]

    def _setup_mock_model(self, mock_model: Mock, mock_processor: Mock) -> Mock:
        """Helper method to setup mock model with proper structure."""
        mock_model_instance = Mock()
        mock_model_instance.config.patch_size = 16
        mock_model_instance.config.num_register_tokens = 0
        mock_model_instance.device = torch.device("cpu")
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval.return_value = mock_model_instance

        # Mock the forward method
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(1, 197, 1024)
        mock_model_instance.return_value = mock_output

        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = self.mock_processor

        # Mock processor to return proper dictionary
        self.mock_processor.return_value = {
            "pixel_values": torch.randn(1, 3, 224, 224),
            "attention_mask": torch.ones(1, 197),
        }

        return mock_model_instance

    @patch("getiprompt.processes.encoders.encoder.optimize_model")
    @patch("getiprompt.processes.encoders.encoder.AutoModel")
    @patch("getiprompt.processes.encoders.encoder.AutoImageProcessor")
    def test_encoder_initialization(self, mock_processor: Mock, mock_model: Mock, mock_optimize: Mock) -> None:
        """Test that encoder initializes correctly."""
        # Setup mocks with proper structure
        mock_model_instance = self._setup_mock_model(mock_model, mock_processor)
        mock_optimize.return_value = mock_model_instance

        # Create encoder
        encoder = Encoder(model_id="dinov2_small", device="cpu", input_size=224)

        # Test initialization
        assert encoder.model == mock_model_instance
        assert encoder.processor == self.mock_processor
        assert encoder.input_size == 224
        assert encoder.patch_size == 16
        assert hasattr(encoder, "mask_transform")

    @patch("getiprompt.processes.encoders.encoder.optimize_model")
    @patch("getiprompt.processes.encoders.encoder.AutoModel")
    @patch("getiprompt.processes.encoders.encoder.AutoImageProcessor")
    def test_mask_transform_creation(self, mock_processor: Mock, mock_model: Mock, mock_optimize: Mock) -> None:
        """Test that mask transform is created correctly."""
        # Setup mocks with proper structure
        mock_model_instance = self._setup_mock_model(mock_model, mock_processor)
        mock_optimize.return_value = mock_model_instance

        # Create encoder
        encoder = Encoder(model_id="dinov2_small", device="cpu", input_size=224)

        # Test that mask transform is a Compose object
        assert hasattr(encoder.mask_transform, "transforms")

        # Test mask transform with dummy data
        dummy_mask = torch.randn(224, 224)
        result = encoder.mask_transform(dummy_mask)

        # Check that result has correct shape
        expected_shape = (1, 14, 14)  # (224/16, 224/16)
        assert result.shape == expected_shape

    @patch("getiprompt.processes.encoders.encoder.optimize_model")
    @patch("getiprompt.processes.encoders.encoder.AutoModel")
    @patch("getiprompt.processes.encoders.encoder.AutoImageProcessor")
    def test_call_without_priors(self, mock_processor: Mock, mock_model: Mock, mock_optimize: Mock) -> None:
        """Test encoder call without priors."""
        # Setup mocks with proper structure
        mock_model_instance = self._setup_mock_model(mock_model, mock_processor)
        mock_optimize.return_value = mock_model_instance

        # Create encoder
        encoder = Encoder(model_id="dinov2_small", device="cpu", input_size=224)

        # Create test data
        images = [Image(np.zeros((224, 224, 3), dtype=np.uint8))]

        # Test encoder call
        features, masks = encoder(images, priors_per_image=None)

        # Check outputs
        assert len(features) == 1
        assert len(masks) == 1
        assert isinstance(features[0], Features)
        assert isinstance(masks[0], Masks)
        assert len(masks[0].data) == 0  # Empty masks

    @staticmethod
    def test_model_id_validation() -> None:
        """Test that invalid model IDs raise ValueError."""
        with pytest.raises(ValueError, match="Invalid model ID"):
            Encoder(model_id="invalid_model")

    @staticmethod
    def test_valid_model_ids() -> None:
        """Test that all valid model IDs are accepted."""
        for model_id in ENCODER_MODEL_COLLECTION:
            with (
                patch("getiprompt.processes.encoders.encoder.optimize_model") as mock_optimize,
                patch("getiprompt.processes.encoders.encoder.AutoModel") as mock_model,
                patch("getiprompt.processes.encoders.encoder.AutoImageProcessor") as mock_processor,
            ):
                # Setup mocks with proper structure
                mock_model_instance = Mock()
                mock_model_instance.config.patch_size = 16
                mock_model_instance.config.num_register_tokens = 0
                mock_model_instance.device = torch.device("cpu")
                mock_model_instance.to.return_value = mock_model_instance
                mock_model_instance.eval.return_value = mock_model_instance

                mock_model.from_pretrained.return_value = mock_model_instance
                mock_processor.from_pretrained.return_value = Mock()
                mock_optimize.return_value = mock_model_instance

                encoder = Encoder(model_id=model_id, device="cpu")
                assert encoder.model_id == model_id

    @patch("getiprompt.processes.encoders.encoder.optimize_model")
    @patch("getiprompt.processes.encoders.encoder.AutoModel")
    @patch("getiprompt.processes.encoders.encoder.AutoImageProcessor")
    def test_encoder_with_different_input_sizes(
        self, mock_processor: Mock, mock_model: Mock, mock_optimize: Mock
    ) -> None:
        """Test encoder with different input sizes."""
        # Setup mocks with proper structure
        mock_model_instance = self._setup_mock_model(mock_model, mock_processor)
        mock_optimize.return_value = mock_model_instance

        # Test with different input sizes
        for input_size in [224, 384, 512]:
            encoder = Encoder(model_id="dinov2_small", device="cpu", input_size=input_size)
            assert encoder.input_size == input_size

    @patch("getiprompt.processes.encoders.encoder.optimize_model")
    @patch("getiprompt.processes.encoders.encoder.AutoModel")
    @patch("getiprompt.processes.encoders.encoder.AutoImageProcessor")
    def test_mask_transform_with_different_sizes(
        self, mock_processor: Mock, mock_model: Mock, mock_optimize: Mock
    ) -> None:
        """Test mask transform with different input sizes."""
        # Setup mocks with proper structure
        mock_model_instance = self._setup_mock_model(mock_model, mock_processor)
        mock_optimize.return_value = mock_model_instance

        # Test with different mask sizes
        for input_size in [224, 384, 512]:
            encoder = Encoder(model_id="dinov2_small", device="cpu", input_size=input_size)

            # Test mask transform
            dummy_mask = torch.randn(input_size, input_size)
            result = encoder.mask_transform(dummy_mask)

            # Check output shape
            expected_size = input_size // 16
            assert result.shape == (1, expected_size, expected_size)

    @patch("getiprompt.processes.encoders.encoder.optimize_model")
    @patch("getiprompt.processes.encoders.encoder.AutoModel")
    @patch("getiprompt.processes.encoders.encoder.AutoImageProcessor")
    def test_encoder_with_priors(self, mock_processor: Mock, mock_model: Mock, mock_optimize: Mock) -> None:
        """Test encoder call with priors."""
        # Setup mocks with proper structure
        mock_model_instance = self._setup_mock_model(mock_model, mock_processor)
        mock_optimize.return_value = mock_model_instance

        # Create encoder
        encoder = Encoder(model_id="dinov2_small", device="cpu", input_size=224)

        # Create test data with priors
        images = [Image(np.zeros((224, 224, 3), dtype=np.uint8))]
        priors = [Priors()]

        # Add a simple mask to priors
        mask = torch.zeros(224, 224)
        mask[50:100, 50:100] = 1  # Small square mask
        priors[0].masks.add(mask=mask, class_id=1)

        # Test encoder call
        features, masks = encoder(images, priors_per_image=priors)

        # Check outputs
        assert len(features) == 1
        assert len(masks) == 1
        assert isinstance(features[0], Features)
        assert isinstance(masks[0], Masks)
        assert len(masks[0].data) > 0  # Should have masks

    @patch("getiprompt.processes.encoders.encoder.optimize_model")
    @patch("getiprompt.processes.encoders.encoder.AutoModel")
    @patch("getiprompt.processes.encoders.encoder.AutoImageProcessor")
    def test_embed_method(self, mock_processor: Mock, mock_model: Mock, mock_optimize: Mock) -> None:
        """Test the _embed method."""
        # Setup mocks with proper structure
        mock_model_instance = self._setup_mock_model(mock_model, mock_processor)
        mock_optimize.return_value = mock_model_instance

        # Create encoder
        encoder = Encoder(model_id="dinov2_small", device="cpu", input_size=224)

        # Create test tensors
        test_tensors = [torch.randn(3, 224, 224)]

        # Mock processor output
        mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

        # Test _embed method
        features = encoder._embed(test_tensors)  # noqa: SLF001

        # Check output
        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == 1  # batch size
        assert features.shape[-1] == 1024  # feature dimension

    @staticmethod
    def test_error_handling_invalid_model_id() -> None:
        """Test error handling for invalid model ID."""
        with pytest.raises(ValueError, match="Invalid model ID"):
            Encoder(model_id="nonexistent_model")

    @staticmethod
    @patch("getiprompt.processes.encoders.encoder.AutoModel")
    def test_huggingface_access_error(mock_model: Mock) -> None:
        """Test error handling for HuggingFace access issues."""
        # Mock OSError for gated repo
        mock_model.from_pretrained.side_effect = OSError("gated repo access denied")

        with pytest.raises(ValueError, match="User does not have access"):
            Encoder(model_id="dinov2_small", device="cpu")


if __name__ == "__main__":
    pytest.main([__file__])
