# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ImageEncoder."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from torchvision.tv_tensors import Image

from getiprompt.components.encoders import AVAILABLE_IMAGE_ENCODERS, ImageEncoder


class TestEncoder:
    """Test the ImageEncoder class."""

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
        """Helper method to setup mock model with proper structure.

        Returns:
            The mock model instance.
        """
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

    @patch("getiprompt.utils.optimization.optimize_model")
    @patch("getiprompt.components.encoders.image_encoder.AutoModel")
    @patch("getiprompt.components.encoders.image_encoder.AutoImageProcessor")
    def test_encoder_initialization(self, mock_processor: Mock, mock_model: Mock, mock_optimize: Mock) -> None:
        """Test that encoder initializes correctly."""
        # Setup mocks with proper structure
        expected_input_size = 224
        expected_patch_size = 16
        mock_model_instance = self._setup_mock_model(mock_model, mock_processor)
        mock_optimize.return_value = mock_model_instance

        # Create encoder
        encoder = ImageEncoder(model_id="dinov2_small", device="cpu", input_size=224)

        # Test initialization
        pytest.assume(encoder.model == mock_model_instance)
        pytest.assume(encoder.processor == self.mock_processor)
        pytest.assume(encoder.input_size == expected_input_size)
        pytest.assume(encoder.patch_size == expected_patch_size)

    @patch("getiprompt.utils.optimization.optimize_model")
    @patch("getiprompt.components.encoders.image_encoder.AutoModel")
    @patch("getiprompt.components.encoders.image_encoder.AutoImageProcessor")
    def test_call_without_priors(self, mock_processor: Mock, mock_model: Mock, mock_optimize: Mock) -> None:
        """Test encoder call without priors."""
        # Setup mocks with proper structure
        mock_model_instance = self._setup_mock_model(mock_model, mock_processor)
        mock_optimize.return_value = mock_model_instance

        # Create encoder
        encoder = ImageEncoder(model_id="dinov2_small", device="cpu", input_size=224)

        # Create test data
        images = [Image(np.zeros((224, 224, 3), dtype=np.uint8))]

        # Test encoder call
        features = encoder(images)

        # Check outputs
        pytest.assume(isinstance(features, torch.Tensor))
        expected_batch_size = 1
        pytest.assume(features.shape[0] == expected_batch_size)

    @staticmethod
    def test_model_id_validation() -> None:
        """Test that invalid model IDs raise ValueError."""
        with pytest.raises(ValueError, match="Invalid model ID"):
            ImageEncoder(model_id="invalid_model")

    @staticmethod
    def test_valid_model_ids() -> None:
        """Test that all valid model IDs are accepted."""
        for model_id in AVAILABLE_IMAGE_ENCODERS:
            with (
                patch("getiprompt.utils.optimization.optimize_model") as mock_optimize,
                patch("getiprompt.components.encoders.image_encoder.AutoModel") as mock_model,
                patch("getiprompt.components.encoders.image_encoder.AutoImageProcessor") as mock_processor,
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

                encoder = ImageEncoder(model_id=model_id, device="cpu")
                pytest.assume(encoder.model_id == model_id)

    @patch("getiprompt.utils.optimization.optimize_model")
    @patch("getiprompt.components.encoders.image_encoder.AutoModel")
    @patch("getiprompt.components.encoders.image_encoder.AutoImageProcessor")
    def test_encoder_with_different_input_sizes(
        self,
        mock_processor: Mock,
        mock_model: Mock,
        mock_optimize: Mock,
    ) -> None:
        """Test encoder with different input sizes."""
        # Setup mocks with proper structure
        mock_model_instance = self._setup_mock_model(mock_model, mock_processor)
        mock_optimize.return_value = mock_model_instance

        # Test with different input sizes
        for input_size in [224, 384, 512]:
            encoder = ImageEncoder(model_id="dinov2_small", device="cpu", input_size=input_size)
            pytest.assume(encoder.input_size == input_size)

    @patch("getiprompt.utils.optimization.optimize_model")
    @patch("getiprompt.components.encoders.image_encoder.AutoModel")
    @patch("getiprompt.components.encoders.image_encoder.AutoImageProcessor")
    def test_encoder_with_different_precisions(
        self,
        mock_processor: Mock,
        mock_model: Mock,
        mock_optimize: Mock,
    ) -> None:
        """Test encoder with different precision settings."""
        # Setup mocks with proper structure
        mock_model_instance = self._setup_mock_model(mock_model, mock_processor)
        mock_optimize.return_value = mock_model_instance

        # Test with different precision settings
        precision_mapping = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        for precision_str, expected_dtype in precision_mapping.items():
            encoder = ImageEncoder(model_id="dinov2_small", device="cpu", precision=precision_str, input_size=224)
            pytest.assume(encoder.precision == expected_dtype)

    @patch("getiprompt.utils.optimization.optimize_model")
    @patch("getiprompt.components.encoders.image_encoder.AutoModel")
    @patch("getiprompt.components.encoders.image_encoder.AutoImageProcessor")
    def test_encoder_with_compile_models(self, mock_processor: Mock, mock_model: Mock, mock_optimize: Mock) -> None:
        """Test encoder with model compilation enabled."""
        # Setup mocks with proper structure
        mock_model_instance = self._setup_mock_model(mock_model, mock_processor)
        mock_optimize.return_value = mock_model_instance

        # Test with compile_models=True
        encoder = ImageEncoder(model_id="dinov2_small", device="cpu", compile_models=True, input_size=224)
        pytest.assume(encoder.model == mock_model_instance)
        # Verify optimize_model was called with compile_models=True
        mock_optimize.assert_called_once()
        call_args = mock_optimize.call_args
        pytest.assume(call_args[1]["compile_models"] is True)

    @patch("getiprompt.utils.optimization.optimize_model")
    @patch("getiprompt.components.encoders.image_encoder.AutoModel")
    @patch("getiprompt.components.encoders.image_encoder.AutoImageProcessor")
    def test_encoder_with_benchmark_inference_speed(
        self,
        mock_processor: Mock,
        mock_model: Mock,
        mock_optimize: Mock,
    ) -> None:
        """Test encoder with benchmark inference speed enabled."""
        # Setup mocks with proper structure
        mock_model_instance = self._setup_mock_model(mock_model, mock_processor)
        mock_optimize.return_value = mock_model_instance

        # Test with benchmark_inference_speed=True
        encoder = ImageEncoder(model_id="dinov2_small", device="cpu", benchmark_inference_speed=True, input_size=224)
        pytest.assume(encoder.model == mock_model_instance)
        # Verify optimize_model was called with benchmark_inference_speed=True
        mock_optimize.assert_called_once()
        call_args = mock_optimize.call_args
        pytest.assume(call_args[1]["benchmark_inference_speed"] is True)

    @patch("getiprompt.utils.optimization.optimize_model")
    @patch("getiprompt.components.encoders.image_encoder.AutoModel")
    @patch("getiprompt.components.encoders.image_encoder.AutoImageProcessor")
    def test_encoder_device_handling(self, mock_processor: Mock, mock_model: Mock, mock_optimize: Mock) -> None:
        """Test encoder device handling."""
        # Setup mocks with proper structure
        mock_model_instance = self._setup_mock_model(mock_model, mock_processor)
        mock_optimize.return_value = mock_model_instance

        # Test with different devices
        for device in ["cpu", "cuda"]:
            encoder = ImageEncoder(model_id="dinov2_small", device=device, input_size=224)
            pytest.assume(encoder.device == device)
            # Verify optimize_model was called with correct device
            mock_optimize.assert_called()
            call_args = mock_optimize.call_args
            pytest.assume(call_args[1]["device"] == device)

    @patch("getiprompt.utils.optimization.optimize_model")
    @patch("getiprompt.components.encoders.image_encoder.AutoModel")
    @patch("getiprompt.components.encoders.image_encoder.AutoImageProcessor")
    def test_encoder_all_parameters(self, mock_processor: Mock, mock_model: Mock, mock_optimize: Mock) -> None:
        """Test encoder with all parameters specified."""
        # Setup mocks with proper structure
        mock_model_instance = self._setup_mock_model(mock_model, mock_processor)
        mock_optimize.return_value = mock_model_instance

        expected_input_size = 384

        # Test with all parameters
        encoder = ImageEncoder(
            model_id="dinov2_base",
            device="cpu",
            precision="fp16",
            compile_models=True,
            benchmark_inference_speed=True,
            input_size=expected_input_size,
        )

        # Verify all parameters are set correctly
        pytest.assume(encoder.model_id == "dinov2_base")
        pytest.assume(encoder.device == "cpu")
        pytest.assume(encoder.precision == torch.float16)
        pytest.assume(encoder.input_size == expected_input_size)
        pytest.assume(encoder.model == mock_model_instance)

        # Verify optimize_model was called with all parameters
        mock_optimize.assert_called_once()
        call_args = mock_optimize.call_args
        pytest.assume(call_args[1]["precision"] == torch.float16)
        pytest.assume(call_args[1]["device"] == "cpu")
        pytest.assume(call_args[1]["compile_models"] is True)
        pytest.assume(call_args[1]["benchmark_inference_speed"] is True)

    @staticmethod
    def test_error_handling_invalid_model_id() -> None:
        """Test error handling for invalid model ID."""
        with pytest.raises(ValueError, match="Invalid model ID"):
            ImageEncoder(model_id="nonexistent_model")

    @staticmethod
    @patch("getiprompt.components.encoders.image_encoder.AutoModel")
    def test_huggingface_access_error(mock_model: Mock) -> None:
        """Test error handling for HuggingFace access issues."""
        # Mock OSError for gated repo
        mock_model.from_pretrained.side_effect = OSError("gated repo access denied")

        with pytest.raises(ValueError, match="User does not have access"):
            ImageEncoder(model_id="dinov2_small", device="cpu")


class TestEncoderIntegration:
    """Integration tests with real DINO models."""

    @pytest.mark.slow
    @pytest.mark.integration
    @staticmethod
    def test_forward_with_real_model_comprehensive() -> None:
        """Comprehensive integration test with real DINOv2 small model."""
        encoder = ImageEncoder(model_id="dinov2_small", device="cpu", input_size=224)

        # Create test image
        rng = np.random.default_rng(42)
        test_image = Image(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))

        # Test forward method
        features = encoder.forward([test_image])

        # Verify outputs
        pytest.assume(isinstance(features, torch.Tensor))
        expected_batch_size = 1
        pytest.assume(features.shape[0] == expected_batch_size)

        # Verify feature shape and properties
        expected_patches = (224 // encoder.patch_size) ** 2
        pytest.assume(features.shape[1] == expected_patches)
        pytest.assume(features.shape[2] == 384)  # dinov2_small has 384 dims

        # Check L2 normalization
        feature_norms = torch.norm(features.float(), dim=-1)
        expected_norms = torch.ones(expected_batch_size, expected_patches, dtype=torch.float32)
        pytest.assume(torch.allclose(feature_norms, expected_norms, atol=1e-2))

    @pytest.mark.slow
    @pytest.mark.integration
    @staticmethod
    def test_model_configuration_validation() -> None:
        """Test that real model configuration is properly loaded."""
        encoder = ImageEncoder(model_id="dinov2_small", device="cpu", input_size=224)

        # Verify model configuration
        expected_patch_size = 14  # DINOv2 small uses 14x14 patches
        expected_feature_size = 16  # 224 / 14 = 16
        expected_ignore_token_length = 5  # CLS token + 4 register tokens

        pytest.assume(encoder.patch_size == expected_patch_size)
        pytest.assume(encoder.feature_size == expected_feature_size)
        pytest.assume(encoder.ignore_token_length == expected_ignore_token_length)

        # Test with different input size
        encoder_384 = ImageEncoder(model_id="dinov2_small", device="cpu", input_size=384)
        pytest.assume(encoder_384.feature_size == 384 // encoder_384.patch_size)

    @pytest.mark.slow
    @pytest.mark.integration
    @staticmethod
    def test_feature_quality_and_consistency() -> None:
        """Test that extracted features are meaningful and consistent."""
        encoder = ImageEncoder(model_id="dinov2_small", device="cpu", input_size=224)

        # Create identical images
        rng = np.random.default_rng(42)
        image_data = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        image1 = Image(image_data.copy())
        image2 = Image(image_data.copy())

        # Extract features
        features1 = encoder.forward([image1])
        features2 = encoder.forward([image2])

        # Features should be identical for identical images
        pytest.assume(torch.allclose(features1, features2, atol=1e-6))

        # Features should be normalized
        feature_norms = torch.norm(features1.float(), dim=-1)
        expected_norms = torch.ones_like(feature_norms, dtype=torch.float32)
        pytest.assume(
            torch.allclose(feature_norms, expected_norms, atol=1e-2),
        )  # Relaxed tolerance for numerical precision

    @pytest.mark.slow
    @pytest.mark.integration
    @staticmethod
    def test_different_model_sizes() -> None:
        """Test with different DINO model sizes to verify configuration."""
        # Test with base model (larger than small)
        encoder_base = ImageEncoder(model_id="dinov2_base", device="cpu", input_size=224)

        # Verify configuration differences
        # Base model has different hidden dimension (768 vs 384)

        # Test feature extraction
        rng = np.random.default_rng(42)
        test_image = Image(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
        features = encoder_base.forward([test_image])

        # Verify feature dimension
        expected_patches = (224 // encoder_base.patch_size) ** 2
        pytest.assume(features.shape == (1, expected_patches, 768))  # batch, patches, 768 dims
