# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified Dino model."""

from unittest.mock import patch

import pytest

from getiprompt.models.dino import ENCODER_MODEL_COLLECTION, Dino


class TestDinoModel:
    """Test the unified Dino model."""

    @staticmethod
    def test_invalid_model_id() -> None:
        """Test that invalid model IDs are rejected."""
        with pytest.raises(ValueError, match="Invalid model ID"):
            Dino(model_id="invalid_model_id")

    @staticmethod
    def test_model_id_mapping() -> None:
        """Test that model IDs are correctly mapped."""
        # Test that all expected model IDs are in the collection
        expected_models = [
            "dinov2_small",
            "dinov2_base",
            "dinov2_large",
            "dinov2_giant",
            "dinov3_small",
            "dinov3_small_plus",
            "dinov3_base",
            "dinov3_large",
            "dinov3_huge",
        ]

        for model_id in expected_models:
            assert model_id in ENCODER_MODEL_COLLECTION

    @staticmethod
    def test_model_id_values() -> None:
        """Test that model ID values are correct."""
        # Test DinoV2 model IDs
        assert ENCODER_MODEL_COLLECTION["dinov2_small"] == "facebook/dinov2-with-registers-small"
        assert ENCODER_MODEL_COLLECTION["dinov2_base"] == "facebook/dinov2-with-registers-base"
        assert ENCODER_MODEL_COLLECTION["dinov2_large"] == "facebook/dinov2-with-registers-large"
        assert ENCODER_MODEL_COLLECTION["dinov2_giant"] == "facebook/dinov2-with-registers-giant"

        # Test DinoV3 model IDs
        assert ENCODER_MODEL_COLLECTION["dinov3_small"] == "facebook/dinov3-vits16-pretrain-lvd1689m"
        assert ENCODER_MODEL_COLLECTION["dinov3_small_plus"] == "facebook/dinov3-vits16plus-pretrain-lvd1689m"
        assert ENCODER_MODEL_COLLECTION["dinov3_base"] == "facebook/dinov3-vitb16-pretrain-lvd1689m"
        assert ENCODER_MODEL_COLLECTION["dinov3_large"] == "facebook/dinov3-vitl16-pretrain-lvd1689m"
        assert ENCODER_MODEL_COLLECTION["dinov3_huge"] == "facebook/dinov3-vith16plus-pretrain-lvd1689m"

    @staticmethod
    def test_default_parameters() -> None:
        """Test that default parameters work correctly."""
        try:
            model = Dino()
            assert model.model_id == "dinov3_large"  # Default model ID
        except ValueError as e:
            # This is expected if model files are not available or user doesn't have access
            if "User does not have access to the weights" in str(e):
                pass
            else:
                raise

    @staticmethod
    def test_custom_model_id() -> None:
        """Test that custom model ID works correctly."""
        try:
            model = Dino(model_id="dinov2_base")
            assert model.model_id == "dinov2_base"
        except ValueError as e:
            # This is expected if model files are not available or user doesn't have access
            if "User does not have access to the weights" in str(e):
                pass
            else:
                raise

    @staticmethod
    def test_huggingface_error_handling() -> None:
        """Test that HuggingFace access errors are handled properly."""
        with patch("getiprompt.models.dino.AutoModel") as mock_model:
            mock_model.from_pretrained.side_effect = OSError("You are trying to access a gated repo.")

            with pytest.raises(ValueError, match="User does not have access to the weights"):
                Dino(model_id="dinov3_large")

    @staticmethod
    def test_model_initialization_parameters() -> None:
        """Test that all initialization parameters work correctly."""
        try:
            model = Dino(
                model_id="dinov2_small",
                device="cpu",
                precision="fp32",
                compile_models=False,
                benchmark_inference_speed=False,
                input_size=224,
            )
            assert model.model_id == "dinov2_small"
            assert model.device == "cpu"
            assert model.input_size == 224
        except ValueError as e:
            # This is expected if model files are not available or user doesn't have access
            if "User does not have access to the weights" in str(e):
                pass
            else:
                raise

    @staticmethod
    def test_model_forward_pass() -> None:
        """Test that the model can perform forward pass."""
        import torch

        try:
            model = Dino(model_id="dinov2_small", device="cpu")
            # Create a dummy input tensor
            dummy_input = torch.randn(1, 3, 224, 224)

            # Test forward pass
            output = model(dummy_input)

            # Check output shape (should be [batch_size, num_patches, feature_dim])
            assert output.shape[0] == 1  # batch size
            assert output.shape[1] > 0  # number of patches
            assert output.shape[2] > 0  # feature dimension

        except ValueError as e:
            # This is expected if model files are not available or user doesn't have access
            if "User does not have access to the weights" in str(e):
                pass
            else:
                raise

    @staticmethod
    def test_model_collection_completeness() -> None:
        """Test that the model collection contains all expected models."""
        # Check that we have both DinoV2 and DinoV3 models
        dinov2_models = [k for k in ENCODER_MODEL_COLLECTION if k.startswith("dinov2_")]
        dinov3_models = [k for k in ENCODER_MODEL_COLLECTION if k.startswith("dinov3_")]

        assert len(dinov2_models) == 4  # small, base, large, giant
        assert len(dinov3_models) == 5  # small, small_plus, base, large, huge

        # Check specific models exist
        assert "dinov2_small" in ENCODER_MODEL_COLLECTION
        assert "dinov2_base" in ENCODER_MODEL_COLLECTION
        assert "dinov2_large" in ENCODER_MODEL_COLLECTION
        assert "dinov2_giant" in ENCODER_MODEL_COLLECTION

        assert "dinov3_small" in ENCODER_MODEL_COLLECTION
        assert "dinov3_small_plus" in ENCODER_MODEL_COLLECTION
        assert "dinov3_base" in ENCODER_MODEL_COLLECTION
        assert "dinov3_large" in ENCODER_MODEL_COLLECTION
        assert "dinov3_huge" in ENCODER_MODEL_COLLECTION


if __name__ == "__main__":
    pytest.main([__file__])
