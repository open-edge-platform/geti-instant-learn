# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DinoEncoder."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from getiprompt.models.dino import Dino
from getiprompt.processes.encoders.dino_encoder import DinoEncoder
from getiprompt.types import Features, Image, Masks


class TestDinoEncoder:
    """Test the DinoEncoder class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Create a mock Dino model
        self.mock_model = Mock()
        self.mock_model.input_size = 224
        self.mock_model.patch_size = 16
        self.mock_model.device = torch.device("cpu")

        # Mock the model's forward method
        self.mock_model.return_value = torch.randn(1, 192, 1024)  # [batch, seq_len, hidden_dim]

        # Create encoder with mock model
        self.encoder = DinoEncoder(self.mock_model)

    def test_encoder_initialization(self) -> None:
        """Test that encoder initializes correctly."""
        assert self.encoder.model == self.mock_model
        assert hasattr(self.encoder, "mask_transform")

    def test_mask_transform_creation(self) -> None:
        """Test that mask transform is created correctly."""
        # Test that mask transform is a Compose object
        assert hasattr(self.encoder.mask_transform, "transforms")

        # Test mask transform with dummy data
        dummy_mask = torch.randn(224, 224)
        result = self.encoder.mask_transform(dummy_mask)

        # Check that result has correct shape
        expected_shape = (1, 14, 14)  # (224/16, 224/16)
        assert result.shape == expected_shape

    def test_call_without_priors(self) -> None:
        """Test encoder call without priors."""
        # Create test data
        images = [Image(np.zeros((224, 224, 3), dtype=np.uint8))]

        # Test encoder call
        features, masks = self.encoder(images, priors_per_image=None)

        # Check outputs
        assert len(features) == 1
        assert len(masks) == 1
        assert isinstance(features[0], Features)
        assert isinstance(masks[0], Masks)
        assert len(masks[0].data) == 0  # Empty masks

    @staticmethod
    def test_encoder_with_different_model_sizes() -> None:
        """Test encoder with different model sizes."""
        # Test with different input sizes
        for input_size in [224, 384, 512]:
            mock_model = Mock()
            mock_model.input_size = input_size
            mock_model.patch_size = 16
            mock_model.device = torch.device("cpu")
            mock_model.return_value = torch.randn(1, 192, 1024)

            encoder = DinoEncoder(mock_model)
            assert encoder.model.input_size == input_size

    @staticmethod
    def test_mask_transform_with_different_sizes() -> None:
        """Test mask transform with different input sizes."""
        # Test with different mask sizes
        for mask_size in [(224, 224), (384, 384), (512, 512)]:
            mock_model = Mock()
            mock_model.input_size = mask_size[0]
            mock_model.patch_size = 16
            mock_model.device = torch.device("cpu")

            encoder = DinoEncoder(mock_model)

            # Test mask transform
            dummy_mask = torch.randn(*mask_size)
            result = encoder.mask_transform(dummy_mask)

            # Check output shape
            expected_size = mask_size[0] // 16
            assert result.shape == (1, expected_size, expected_size)

    @staticmethod
    def test_encoder_with_real_dino_model() -> None:
        """Test encoder with a real Dino model."""
        # Create a real Dino model (small version for testing)
        dino_model = Dino(version="v2", size="small", device="cpu")

        # Create encoder
        encoder = DinoEncoder(dino_model)

        # Test that encoder has the model
        assert encoder.model == dino_model

        # Test actual model properties
        assert encoder.model.input_size == dino_model.input_size
        assert encoder.model.patch_size == dino_model.patch_size

        # Test with a real image
        rng = np.random.default_rng()
        input_size = dino_model.input_size
        test_image = Image(rng.integers(0, 255, (input_size, input_size, 3), dtype=np.uint8))
        features, masks = encoder([test_image], priors_per_image=None)

        # Verify outputs
        assert len(features) == 1
        assert len(masks) == 1
        assert isinstance(features[0], Features)
        assert isinstance(masks[0], Masks)

    def test_error_handling(self) -> None:
        """Test error handling in encoder."""
        # Test with invalid input
        with pytest.raises(AttributeError):
            # This should fail because the mock model doesn't have the right attributes
            self.encoder._extract_global_features_batch([None])  # noqa: SLF001


if __name__ == "__main__":
    pytest.main([__file__])
