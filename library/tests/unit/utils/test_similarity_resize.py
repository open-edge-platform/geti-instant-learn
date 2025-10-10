# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for similarity map resizing utilities."""

import pytest
import torch

from getiprompt.utils.similarity_resize import resize_similarity_map


class TestResizeSimilarityMap:
    """Test cases for resize_similarity_map function."""

    def test_basic_resize_square_to_square(self) -> None:
        """Test basic resize from square to square dimensions."""
        # 64x64 flattened to 4096
        similarities = torch.randn(1, 4096)
        resized = resize_similarity_map(similarities, target_size=(128, 128))

        assert resized.shape == (128, 128)

    def test_resize_to_rectangular(self) -> None:
        """Test resize to non-square target size."""
        # 16x16 flattened to 256
        similarities = torch.randn(2, 256)
        resized = resize_similarity_map(similarities, target_size=(64, 128))

        assert resized.shape == (2, 64, 128)

    def test_resize_with_single_int_target(self) -> None:
        """Test resize with single integer as target size."""
        similarities = torch.randn(1, 1024)  # 32x32
        resized = resize_similarity_map(similarities, target_size=256)

        assert resized.shape == (256, 256)

    def test_unpadded_image_size_square(self) -> None:
        """Test padding removal with square unpadded_image_size."""
        # 64x64 flattened
        similarities = torch.randn(1, 4096)
        resized = resize_similarity_map(
            similarities,
            target_size=(256, 256),
            unpadded_image_size=(240, 240),
        )

        assert resized.shape == (256, 256)

    def test_unpadded_image_size_rectangular(self) -> None:
        """Test padding removal with non-square unpadded size."""
        similarities = torch.randn(1, 4096)  # 64x64
        resized = resize_similarity_map(
            similarities,
            target_size=(200, 300),
            unpadded_image_size=(180, 280),
        )

        assert resized.shape == (200, 300)

    def test_batch_size_greater_than_one(self) -> None:
        """Test with batch size > 1."""
        similarities = torch.randn(4, 1024)  # batch=4, 32x32
        resized = resize_similarity_map(similarities, target_size=(64, 64))

        assert resized.shape == (4, 64, 64)

    def test_single_batch_gets_squeezed(self) -> None:
        """Test that single batch dimension gets squeezed."""
        similarities = torch.randn(1, 1024)  # 32x32
        resized = resize_similarity_map(similarities, target_size=(64, 64))

        # Should squeeze to 2D if batch=1
        assert resized.ndim == 2
        assert resized.shape == (64, 64)

    def test_preserves_values_order_of_magnitude(self) -> None:
        """Test that resizing preserves approximate value ranges."""
        similarities = torch.ones(1, 256) * 0.5  # 16x16, all 0.5
        resized = resize_similarity_map(similarities, target_size=(32, 32))

        # Bilinear interpolation should preserve approximate values
        assert torch.allclose(resized, torch.ones(32, 32) * 0.5, atol=0.1)

    def test_dtype_preservation(self) -> None:
        """Test that data type is preserved."""
        similarities_fp32 = torch.randn(1, 1024, dtype=torch.float32)
        resized_fp32 = resize_similarity_map(similarities_fp32, target_size=(64, 64))
        assert resized_fp32.dtype == torch.float32

        similarities_fp16 = torch.randn(1, 1024, dtype=torch.float16)
        resized_fp16 = resize_similarity_map(similarities_fp16, target_size=(64, 64))
        assert resized_fp16.dtype == torch.float16

    def test_device_preservation(self) -> None:
        """Test that device is preserved."""
        similarities = torch.randn(1, 1024)
        resized = resize_similarity_map(similarities, target_size=(64, 64))
        assert resized.device == similarities.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self) -> None:
        """Test with CUDA tensors."""
        similarities = torch.randn(1, 1024, device="cuda")
        resized = resize_similarity_map(similarities, target_size=(64, 64))
        assert resized.device.type == "cuda"
        assert resized.shape == (64, 64)

    def test_zero_tensor(self) -> None:
        """Test with all-zero tensor."""
        similarities = torch.zeros(1, 1024)
        resized = resize_similarity_map(similarities, target_size=(64, 64))
        assert torch.allclose(resized, torch.zeros(64, 64))

    def test_large_batch(self) -> None:
        """Test with large batch size."""
        similarities = torch.randn(16, 256)  # batch=16, 16x16
        resized = resize_similarity_map(similarities, target_size=(32, 32))
        assert resized.shape == (16, 32, 32)

    def test_upscaling(self) -> None:
        """Test upscaling from smaller to larger size."""
        similarities = torch.randn(1, 64)  # 8x8
        resized = resize_similarity_map(similarities, target_size=(256, 256))
        assert resized.shape == (256, 256)

    def test_downscaling(self) -> None:
        """Test downscaling from larger to smaller size."""
        similarities = torch.randn(1, 16384)  # 128x128
        resized = resize_similarity_map(similarities, target_size=(32, 32))
        assert resized.shape == (32, 32)
