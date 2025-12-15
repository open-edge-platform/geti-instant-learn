# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ImageEncoder."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from torchvision.tv_tensors import Image

from getiprompt.components.encoders import (
    AVAILABLE_IMAGE_ENCODERS,
    TIMM_AVAILABLE_IMAGE_ENCODERS,
    ImageEncoder,
)
from getiprompt.utils.constants import Backend


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

    def _setup_mock_hf_model(self, mock_model: Mock, mock_processor: Mock) -> Mock:
        """Helper method to setup mock HuggingFace model with proper structure.

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

    def _setup_mock_timm_model(self, mock_timm_create: Mock, mock_data_config: Mock) -> Mock:
        """Helper method to setup mock TIMM model with proper structure.

        Returns:
            The mock model instance.
        """
        mock_model_instance = Mock()
        mock_model_instance.patch_embed.patch_size = [16, 16]
        mock_model_instance.num_prefix_tokens = 1
        mock_model_instance.device = torch.device("cpu")
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval.return_value = mock_model_instance

        # Mock forward_features method
        mock_model_instance.forward_features.return_value = torch.randn(1, 197, 1024)

        mock_timm_create.return_value = mock_model_instance

        # Mock data config
        mock_data_config.return_value = {
            "input_size": (3, 512, 512),
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }

        return mock_model_instance

    @pytest.mark.parametrize("backend", [Backend.HUGGINGFACE, Backend.TIMM])
    @patch("getiprompt.utils.optimization.optimize_model")
    @patch("getiprompt.components.encoders.timm.timm.create_model")
    @patch("getiprompt.components.encoders.timm.timm.data.resolve_model_data_config")
    @patch("getiprompt.components.encoders.huggingface.AutoModel")
    @patch("getiprompt.components.encoders.huggingface.AutoImageProcessor")
    def test_encoder_initialization(
        self,
        mock_processor: Mock,
        mock_model: Mock,
        mock_timm_data_config: Mock,
        mock_timm_create: Mock,
        mock_optimize: Mock,
        backend: Backend,
    ) -> None:
        """Test that encoder initializes correctly for both backends."""
        expected_input_size = 224
        expected_patch_size = 16
        mock_optimize.return_value = Mock()

        if backend == Backend.HUGGINGFACE:
            mock_model_instance = self._setup_mock_hf_model(mock_model, mock_processor)
            mock_optimize.return_value = mock_model_instance
            model_id = "dinov2_small"
        else:  # TIMM
            mock_model_instance = self._setup_mock_timm_model(mock_timm_create, mock_timm_data_config)
            mock_optimize.return_value = mock_model_instance
            model_id = "dinov3_small"

        # Create encoder
        encoder = ImageEncoder(model_id=model_id, backend=backend, device="cpu", input_size=expected_input_size)

        # Test initialization
        pytest.assume(encoder._model.model == mock_model_instance)
        pytest.assume(encoder.input_size == expected_input_size)
        pytest.assume(encoder.patch_size == expected_patch_size)

    @pytest.mark.parametrize("backend", [Backend.HUGGINGFACE, Backend.TIMM])
    @patch("getiprompt.utils.optimization.optimize_model")
    @patch("getiprompt.components.encoders.timm.timm.create_model")
    @patch("getiprompt.components.encoders.timm.timm.data.resolve_model_data_config")
    @patch("getiprompt.components.encoders.huggingface.AutoModel")
    @patch("getiprompt.components.encoders.huggingface.AutoImageProcessor")
    def test_call_without_priors(
        self,
        mock_processor: Mock,
        mock_model: Mock,
        mock_timm_data_config: Mock,
        mock_timm_create: Mock,
        mock_optimize: Mock,
        backend: Backend,
    ) -> None:
        """Test encoder call without priors for both backends."""
        if backend == Backend.HUGGINGFACE:
            mock_model_instance = self._setup_mock_hf_model(mock_model, mock_processor)
            mock_optimize.return_value = mock_model_instance
            model_id = "dinov2_small"
        else:  # TIMM
            mock_model_instance = self._setup_mock_timm_model(mock_timm_create, mock_timm_data_config)
            mock_optimize.return_value = mock_model_instance
            model_id = "dinov3_small"

        # Create encoder
        encoder = ImageEncoder(model_id=model_id, backend=backend, device="cpu", input_size=224)

        # Create test data
        images = [Image(np.zeros((3, 224, 224), dtype=np.uint8))]

        # Test encoder call
        embeddings = encoder(images)

        # Check outputs
        pytest.assume(isinstance(embeddings, torch.Tensor))
        expected_batch_size = 1
        pytest.assume(embeddings.shape[0] == expected_batch_size)

    @pytest.mark.parametrize("backend", [Backend.HUGGINGFACE, Backend.TIMM])
    @staticmethod
    def test_model_id_validation(backend: Backend) -> None:
        """Test that invalid model IDs raise ValueError for both backends."""
        with pytest.raises(ValueError, match="Invalid model ID"):
            ImageEncoder(model_id="invalid_model", backend=backend)

    @staticmethod
    def test_valid_model_ids_huggingface() -> None:
        """Test that all valid HuggingFace model IDs are accepted."""
        for model_id in AVAILABLE_IMAGE_ENCODERS:
            with (
                patch("getiprompt.utils.optimization.optimize_model") as mock_optimize,
                patch("getiprompt.components.encoders.huggingface.AutoModel") as mock_model,
                patch("getiprompt.components.encoders.huggingface.AutoImageProcessor") as mock_processor,
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

                encoder = ImageEncoder(model_id=model_id, backend=Backend.HUGGINGFACE, device="cpu")
                pytest.assume(encoder.model_id == model_id)

    @staticmethod
    def test_valid_model_ids_timm() -> None:
        """Test that all valid TIMM model IDs are accepted."""
        for model_id in TIMM_AVAILABLE_IMAGE_ENCODERS:
            with (
                patch("getiprompt.utils.optimization.optimize_model") as mock_optimize,
                patch("getiprompt.components.encoders.timm.timm.create_model") as mock_timm_create,
                patch(
                    "getiprompt.components.encoders.timm.timm.data.resolve_model_data_config",
                ) as mock_timm_data_config,
            ):
                # Setup mocks with proper structure
                mock_model_instance = Mock()
                mock_model_instance.patch_embed.patch_size = [16, 16]
                mock_model_instance.num_prefix_tokens = 1
                mock_model_instance.device = torch.device("cpu")
                mock_model_instance.to.return_value = mock_model_instance
                mock_model_instance.eval.return_value = mock_model_instance

                mock_timm_create.return_value = mock_model_instance
                mock_timm_data_config.return_value = {
                    "input_size": (3, 512, 512),
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                }
                mock_optimize.return_value = mock_model_instance

                encoder = ImageEncoder(model_id=model_id, backend=Backend.TIMM, device="cpu")
                pytest.assume(encoder.model_id == model_id)

    @pytest.mark.parametrize("backend", [Backend.HUGGINGFACE, Backend.TIMM])
    @patch("getiprompt.utils.optimization.optimize_model")
    @patch("getiprompt.components.encoders.timm.timm.create_model")
    @patch("getiprompt.components.encoders.timm.timm.data.resolve_model_data_config")
    @patch("getiprompt.components.encoders.huggingface.AutoModel")
    @patch("getiprompt.components.encoders.huggingface.AutoImageProcessor")
    def test_encoder_with_different_input_sizes(
        self,
        mock_processor: Mock,
        mock_model: Mock,
        mock_timm_data_config: Mock,
        mock_timm_create: Mock,
        mock_optimize: Mock,
        backend: Backend,
    ) -> None:
        """Test encoder with different input sizes for both backends."""
        if backend == Backend.HUGGINGFACE:
            mock_model_instance = self._setup_mock_hf_model(mock_model, mock_processor)
            mock_optimize.return_value = mock_model_instance
            model_id = "dinov2_small"
        else:  # TIMM
            mock_model_instance = self._setup_mock_timm_model(mock_timm_create, mock_timm_data_config)
            mock_optimize.return_value = mock_model_instance
            model_id = "dinov3_small"

        # Test with different input sizes
        for input_size in [224, 384, 512]:
            encoder = ImageEncoder(model_id=model_id, backend=backend, device="cpu", input_size=input_size)
            pytest.assume(encoder.input_size == input_size)

    @pytest.mark.parametrize("backend", [Backend.HUGGINGFACE, Backend.TIMM])
    @patch("getiprompt.utils.optimization.optimize_model")
    @patch("getiprompt.components.encoders.timm.timm.create_model")
    @patch("getiprompt.components.encoders.timm.timm.data.resolve_model_data_config")
    @patch("getiprompt.components.encoders.huggingface.AutoModel")
    @patch("getiprompt.components.encoders.huggingface.AutoImageProcessor")
    def test_encoder_with_different_precisions(
        self,
        mock_processor: Mock,
        mock_model: Mock,
        mock_timm_data_config: Mock,
        mock_timm_create: Mock,
        mock_optimize: Mock,
        backend: Backend,
    ) -> None:
        """Test encoder with different precision settings for both backends."""
        if backend == Backend.HUGGINGFACE:
            mock_model_instance = self._setup_mock_hf_model(mock_model, mock_processor)
            mock_optimize.return_value = mock_model_instance
            model_id = "dinov2_small"
        else:  # TIMM
            mock_model_instance = self._setup_mock_timm_model(mock_timm_create, mock_timm_data_config)
            mock_optimize.return_value = mock_model_instance
            model_id = "dinov3_small"

        # Test with different precision settings
        precision_mapping = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        for precision_str, expected_dtype in precision_mapping.items():
            encoder = ImageEncoder(
                model_id=model_id,
                backend=backend,
                device="cpu",
                precision=precision_str,
                input_size=224,
            )
            pytest.assume(encoder._model.precision == expected_dtype)

    @pytest.mark.parametrize("backend", [Backend.HUGGINGFACE, Backend.TIMM])
    @patch("getiprompt.utils.optimization.optimize_model")
    @patch("getiprompt.components.encoders.timm.timm.create_model")
    @patch("getiprompt.components.encoders.timm.timm.data.resolve_model_data_config")
    @patch("getiprompt.components.encoders.huggingface.AutoModel")
    @patch("getiprompt.components.encoders.huggingface.AutoImageProcessor")
    def test_encoder_with_compile_models(
        self,
        mock_processor: Mock,
        mock_model: Mock,
        mock_timm_data_config: Mock,
        mock_timm_create: Mock,
        mock_optimize: Mock,
        backend: Backend,
    ) -> None:
        """Test encoder with model compilation enabled for both backends."""
        if backend == Backend.HUGGINGFACE:
            mock_model_instance = self._setup_mock_hf_model(mock_model, mock_processor)
            mock_optimize.return_value = mock_model_instance
            model_id = "dinov2_small"
        else:  # TIMM
            mock_model_instance = self._setup_mock_timm_model(mock_timm_create, mock_timm_data_config)
            mock_optimize.return_value = mock_model_instance
            model_id = "dinov3_small"

        # Test with compile_models=True
        encoder = ImageEncoder(model_id=model_id, backend=backend, device="cpu", compile_models=True, input_size=224)
        pytest.assume(encoder._model.model == mock_model_instance)
        # Verify optimize_model was called with compile_models=True
        mock_optimize.assert_called_once()
        call_args = mock_optimize.call_args
        pytest.assume(call_args[1]["compile_models"] is True)

    @pytest.mark.parametrize("backend", [Backend.HUGGINGFACE, Backend.TIMM])
    @patch("getiprompt.utils.optimization.optimize_model")
    @patch("getiprompt.components.encoders.timm.timm.create_model")
    @patch("getiprompt.components.encoders.timm.timm.data.resolve_model_data_config")
    @patch("getiprompt.components.encoders.huggingface.AutoModel")
    @patch("getiprompt.components.encoders.huggingface.AutoImageProcessor")
    def test_encoder_device_handling(
        self,
        mock_processor: Mock,
        mock_model: Mock,
        mock_timm_data_config: Mock,
        mock_timm_create: Mock,
        mock_optimize: Mock,
        backend: Backend,
    ) -> None:
        """Test encoder device handling for both backends."""
        if backend == Backend.HUGGINGFACE:
            mock_model_instance = self._setup_mock_hf_model(mock_model, mock_processor)
            mock_optimize.return_value = mock_model_instance
            model_id = "dinov2_small"
        else:  # TIMM
            mock_model_instance = self._setup_mock_timm_model(mock_timm_create, mock_timm_data_config)
            mock_optimize.return_value = mock_model_instance
            model_id = "dinov3_small"

        # Test with different devices
        for device in ["cpu", "cuda"]:
            encoder = ImageEncoder(model_id=model_id, backend=backend, device=device, input_size=224)
            pytest.assume(encoder.device == device)
            # Verify optimize_model was called with correct device
            mock_optimize.assert_called()
            call_args = mock_optimize.call_args
            pytest.assume(call_args[1]["device"] == device)

    @staticmethod
    def test_error_handling_invalid_model_id() -> None:
        """Test error handling for invalid model ID."""
        with pytest.raises(ValueError, match="Invalid model ID"):
            ImageEncoder(model_id="nonexistent_model")

    @staticmethod
    @patch("getiprompt.components.encoders.huggingface.AutoModel")
    def test_huggingface_access_error(mock_model: Mock) -> None:
        """Test error handling for HuggingFace access issues."""
        # Mock OSError for gated repo
        mock_model.from_pretrained.side_effect = OSError("gated repo access denied")

        with pytest.raises(ValueError, match="User does not have access"):
            ImageEncoder(model_id="dinov2_small", backend=Backend.HUGGINGFACE, device="cpu")


class TestEncoderIntegration:
    """Integration tests with real DINO models."""

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.parametrize("backend", [Backend.HUGGINGFACE, Backend.TIMM])
    @staticmethod
    def test_forward_with_real_model_comprehensive(backend: Backend) -> None:
        """Comprehensive integration test with real DINO models for both backends."""
        if backend == Backend.HUGGINGFACE:
            model_id = "dinov2_small"
        else:  # TIMM
            model_id = "dinov3_small"
        encoder = ImageEncoder(model_id=model_id, backend=backend, device="cpu", input_size=224)

        # Create test image
        rng = np.random.default_rng(42)
        test_image = Image(rng.integers(0, 255, (3, 224, 224), dtype=np.uint8))

        # Test forward method
        embeddings = encoder.forward([test_image])

        # Verify outputs
        pytest.assume(isinstance(embeddings, torch.Tensor))
        expected_batch_size = 1
        pytest.assume(embeddings.shape[0] == expected_batch_size)

        # Verify feature shape and properties
        expected_patches = (224 // encoder.patch_size) ** 2
        pytest.assume(embeddings.shape[1] == expected_patches)
        # Embedding dimension depends on model
        if backend == Backend.HUGGINGFACE:
            pytest.assume(embeddings.shape[2] == 384)  # dinov2_small has 384 dims
        else:  # TIMM
            pytest.assume(embeddings.shape[2] > 0)  # Just check it's positive

        # Check L2 normalization
        feature_norms = torch.norm(embeddings.float(), dim=-1)
        expected_norms = torch.ones(expected_batch_size, expected_patches, dtype=torch.float32)
        pytest.assume(torch.allclose(feature_norms, expected_norms, atol=1e-2))

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.parametrize("backend", [Backend.HUGGINGFACE, Backend.TIMM])
    @staticmethod
    def test_model_configuration_validation(backend: Backend) -> None:
        """Test that real model configuration is properly loaded for both backends."""
        expected_ignore_token_length = 5  # CLS token only
        if backend == Backend.HUGGINGFACE:
            model_id = "dinov2_small"
            expected_patch_size = 14  # DINOv2 small uses 14x14 patches
        else:  # TIMM
            model_id = "dinov3_small"
            expected_patch_size = 16  # DINOv3 uses 16x16 patches

        encoder = ImageEncoder(model_id=model_id, backend=backend, device="cpu", input_size=224)
        expected_feature_size = 224 // expected_patch_size

        pytest.assume(encoder.patch_size == expected_patch_size)
        pytest.assume(encoder.feature_size == expected_feature_size)
        pytest.assume(encoder._model.ignore_token_length == expected_ignore_token_length)

        # Test with different input size
        encoder_384 = ImageEncoder(model_id=model_id, backend=backend, device="cpu", input_size=384)
        pytest.assume(encoder_384.feature_size == 384 // encoder_384.patch_size)

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.parametrize("backend", [Backend.HUGGINGFACE, Backend.TIMM])
    @staticmethod
    def test_feature_quality_and_consistency(backend: Backend) -> None:
        """Test that extracted embeddings are meaningful and consistent for both backends."""
        if backend == Backend.HUGGINGFACE:
            model_id = "dinov2_small"
        else:  # TIMM
            model_id = "dinov3_small"
        encoder = ImageEncoder(model_id=model_id, backend=backend, device="cpu", input_size=224)

        # Create identical images
        rng = np.random.default_rng(42)
        image_data = rng.integers(0, 255, (3, 224, 224), dtype=np.uint8)
        image1 = Image(image_data.copy())
        image2 = Image(image_data.copy())

        # Extract embeddings
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
    @pytest.mark.parametrize("backend", [Backend.HUGGINGFACE, Backend.TIMM])
    @staticmethod
    def test_different_model_sizes(backend: Backend) -> None:
        """Test with different DINO model sizes to verify configuration for both backends."""
        if backend == Backend.HUGGINGFACE:
            model_id = "dinov2_base"
        else:  # TIMM
            model_id = "dinov3_base"
        # Test with base model (larger than small)
        encoder_base = ImageEncoder(model_id=model_id, backend=backend, device="cpu", input_size=224)

        # Verify configuration differences
        # Base model has different hidden dimension (768 vs 384)

        # Test feature extraction
        rng = np.random.default_rng(42)
        test_image = Image(rng.integers(0, 255, (3, 224, 224), dtype=np.uint8))
        embeddings = encoder_base.forward([test_image])

        # Verify feature dimension
        expected_patches = (224 // encoder_base.patch_size) ** 2
        # Embedding dimension depends on model and backend
        if backend == Backend.HUGGINGFACE:
            pytest.assume(embeddings.shape == (1, expected_patches, 768))  # batch, patches, 768 dims
        else:  # TIMM
            pytest.assume(embeddings.shape[0] == 1)
            pytest.assume(embeddings.shape[1] == expected_patches)
            pytest.assume(embeddings.shape[2] > 0)  # Just check it's positive
