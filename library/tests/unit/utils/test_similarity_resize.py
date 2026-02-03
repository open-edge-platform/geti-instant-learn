# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for similarity map resizing utilities."""

import pytest
import torch

from instantlearn.utils.similarity_resize import resize_similarity_maps


class TestResizeSimilarityMapBasic:
    """Test basic resize functionality."""

    def test_basic_resize_square_to_square(self) -> None:
        """Test basic resize from square to square dimensions."""
        # 64x64 flattened to 4096
        similarities = torch.randn(1, 4096)
        resized = resize_similarity_maps(similarities, target_size=(128, 128))

        expected_shape = (1, 128, 128)
        actual_shape = resized.shape
        pytest.assume(actual_shape == expected_shape)

    def test_resize_to_rectangular(self) -> None:
        """Test resize to non-square target size."""
        # 16x16 flattened to 256
        similarities = torch.randn(2, 256)
        resized = resize_similarity_maps(similarities, target_size=(64, 128))

        expected_shape = (2, 64, 128)
        actual_shape = resized.shape
        pytest.assume(actual_shape == expected_shape)

    def test_resize_with_single_int_target(self) -> None:
        """Test resize with single integer as target size."""
        similarities = torch.randn(1, 1024)  # 32x32
        resized = resize_similarity_maps(similarities, target_size=256)

        expected_shape = (1, 256, 256)
        actual_shape = resized.shape
        pytest.assume(actual_shape == expected_shape)

    def test_non_square_input_to_square_output(self) -> None:
        """Test resizing non-square input to square output."""
        similarities = torch.randn(1, 3600)  # 60x60
        resized = resize_similarity_maps(similarities, target_size=(100, 100))
        pytest.assume(resized.shape == (1, 100, 100))

    def test_upscaling(self) -> None:
        """Test upscaling from smaller to larger size."""
        similarities = torch.randn(1, 64)  # 8x8
        resized = resize_similarity_maps(similarities, target_size=(256, 256))
        pytest.assume(resized.shape == (1, 256, 256))

    def test_downscaling(self) -> None:
        """Test downscaling from larger to smaller size."""
        similarities = torch.randn(1, 16384)  # 128x128
        resized = resize_similarity_maps(similarities, target_size=(32, 32))
        pytest.assume(resized.shape == (1, 32, 32))

    def test_identical_input_output_size(self) -> None:
        """Test when input and output sizes are identical."""
        similarities = torch.randn(1, 4096)  # 64x64
        resized = resize_similarity_maps(similarities, target_size=(64, 64))
        pytest.assume(resized.shape == (1, 64, 64))


class TestResizeSimilarityMapBatching:
    """Test batch processing functionality."""

    def test_batch_size_greater_than_one(self) -> None:
        """Test with batch size > 1."""
        similarities = torch.randn(4, 1024)  # batch=4, 32x32
        resized = resize_similarity_maps(similarities, target_size=(64, 64))

        expected_shape = (4, 64, 64)
        actual_shape = resized.shape
        pytest.assume(actual_shape == expected_shape)

    def test_single_batch_gets_squeezed(self) -> None:
        """Test that single batch dimension gets squeezed."""
        similarities = torch.randn(1, 1024)  # 32x32
        resized = resize_similarity_maps(similarities, target_size=(64, 64))

        # Should squeeze to 2D if batch=1
        expected_dim = 3
        pytest.assume(resized.ndim == expected_dim)
        pytest.assume(resized.shape == (1, 64, 64))

    def test_large_batch(self) -> None:
        """Test with large batch size."""
        similarities = torch.randn(16, 256)  # batch=16, 16x16
        resized = resize_similarity_maps(similarities, target_size=(32, 32))
        pytest.assume(resized.shape == (16, 32, 32))


class TestResizeSimilarityMapDataTypes:
    """Test data type and device preservation."""

    def test_dtype_preservation(self) -> None:
        """Test that data type is preserved."""
        similarities_fp32 = torch.randn(1, 1024, dtype=torch.float32)
        resized_fp32 = resize_similarity_maps(similarities_fp32, target_size=(64, 64))
        pytest.assume(resized_fp32.dtype == torch.float32)

        similarities_fp16 = torch.randn(1, 1024, dtype=torch.float16)
        resized_fp16 = resize_similarity_maps(similarities_fp16, target_size=(64, 64))
        pytest.assume(resized_fp16.dtype == torch.float16)

    def test_high_precision_dtype(self) -> None:
        """Test with double precision tensors."""
        similarities = torch.randn(1, 1024, dtype=torch.float64)
        resized = resize_similarity_maps(similarities, target_size=(64, 64))
        pytest.assume(resized.dtype == torch.float64)

    def test_device_preservation(self) -> None:
        """Test that device is preserved."""
        similarities = torch.randn(1, 1024)
        resized = resize_similarity_maps(similarities, target_size=(64, 64))
        pytest.assume(resized.device == similarities.device)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self) -> None:
        """Test with CUDA tensors."""
        similarities = torch.randn(1, 1024, device="cuda")
        resized = resize_similarity_maps(similarities, target_size=(64, 64))
        pytest.assume(resized.device.type == "cuda")
        pytest.assume(resized[0].shape == (64, 64))


class TestResizeSimilarityMapValuePreservation:
    """Test value and gradient preservation."""

    def test_preserves_values_order_of_magnitude(self) -> None:
        """Test that resizing preserves approximate value ranges."""
        similarities = torch.ones(1, 256) * 0.5  # 16x16, all 0.5
        resized = resize_similarity_maps(similarities, target_size=(32, 32))

        # Bilinear interpolation should preserve approximate values
        expected = torch.ones(32, 32) * 0.5
        pytest.assume(torch.allclose(resized, expected, atol=0.1))

    def test_zero_tensor(self) -> None:
        """Test with all-zero tensor."""
        similarities = torch.zeros(1, 1024)
        resized = resize_similarity_maps(similarities, target_size=(64, 64))
        expected = torch.zeros(64, 64)
        pytest.assume(torch.allclose(resized, expected))

    def test_preserves_gradient_flow(self) -> None:
        """Test that gradients can flow through the resize operation."""
        similarities = torch.randn(1, 256, requires_grad=True)
        resized = resize_similarity_maps(similarities, target_size=(32, 32))
        pytest.assume(resized.requires_grad)


class TestResizeSimilarityMapEdgeCases:
    """Test edge cases and extreme scenarios."""

    def test_extreme_upscaling(self) -> None:
        """Test extreme upscaling factor."""
        similarities = torch.randn(1, 16)  # 4x4
        resized = resize_similarity_maps(similarities, target_size=(512, 512))
        pytest.assume(resized.shape == (1, 512, 512))

    def test_extreme_downscaling(self) -> None:
        """Test extreme downscaling factor."""
        similarities = torch.randn(1, 65536)  # 256x256
        resized = resize_similarity_maps(similarities, target_size=(16, 16))
        pytest.assume(resized.shape == (1, 16, 16))

    def test_single_pixel_target(self) -> None:
        """Test resize to single pixel."""
        similarities = torch.randn(1, 1024)  # 32x32
        resized = resize_similarity_maps(similarities, target_size=(1, 1))
        pytest.assume(resized.shape == (1, 1, 1))
