# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for image and mask resizing utilities."""

import pytest
import torch

from instantlearn.utils.image_resize import DownscaleResult, downscale_image, upscale_masks


class TestDownscaleImage:
    def test_no_downscale_when_within_limit(self) -> None:
        """Image smaller than max_side is returned unchanged."""
        image = torch.randn(3, 512, 768)
        result = downscale_image(image, max_side=1024)

        assert result.image is image
        assert result.scale == 1.0
        assert result.orig_size == (512, 768)
        assert not result.is_scaled

    def test_no_downscale_when_equal_to_limit(self) -> None:
        """Image exactly at max_side is returned unchanged."""
        image = torch.randn(3, 1024, 800)
        result = downscale_image(image, max_side=1024)

        assert result.image is image
        assert result.scale == 1.0
        assert not result.is_scaled

    def test_downscale_landscape(self) -> None:
        """Landscape image is downscaled based on width."""
        image = torch.randn(3, 1000, 2000)
        result = downscale_image(image, max_side=1000)

        assert result.is_scaled
        assert result.scale == pytest.approx(0.5)
        assert result.orig_size == (1000, 2000)
        assert result.image.shape == (3, 500, 1000)

    def test_downscale_portrait(self) -> None:
        """Portrait image is downscaled based on height."""
        image = torch.randn(3, 2000, 1000)
        result = downscale_image(image, max_side=1000)

        assert result.is_scaled
        assert result.scale == pytest.approx(0.5)
        assert result.image.shape == (3, 1000, 500)

    def test_downscale_preserves_aspect_ratio(self) -> None:
        """Aspect ratio is preserved after downscaling."""
        image = torch.randn(3, 2776, 2082)
        result = downscale_image(image, max_side=1024)

        original_ratio = 2082 / 2776
        new_h, new_w = result.image.shape[1], result.image.shape[2]
        new_ratio = new_w / new_h

        assert new_ratio == pytest.approx(original_ratio, abs=0.01)

    def test_downscale_preserves_dtype(self) -> None:
        """Output tensor dtype matches input dtype."""
        for dtype in [torch.float32, torch.bfloat16, torch.uint8]:
            image = torch.zeros(3, 2048, 2048, dtype=dtype)
            result = downscale_image(image, max_side=1024)
            assert result.image.dtype == dtype

    def test_downscale_longest_side_within_limit(self) -> None:
        """Downscaled image longest side does not exceed max_side."""
        image = torch.randn(3, 3000, 4000)
        result = downscale_image(image, max_side=1024)

        assert max(result.image.shape[1], result.image.shape[2]) <= 1024


class TestUpscaleMasks:
    def test_upscale_to_original_size(self) -> None:
        """Masks are upscaled to the requested size."""
        masks = torch.ones(5, 50, 100, dtype=torch.bool)
        result = upscale_masks(masks, orig_size=(200, 400))

        assert result.shape == (5, 200, 400)
        assert result.dtype == torch.bool

    def test_upscale_preserves_true_values(self) -> None:
        """All-true masks remain all-true after upscale."""
        masks = torch.ones(2, 64, 64, dtype=torch.bool)
        result = upscale_masks(masks, orig_size=(256, 256))

        assert result.all()

    def test_upscale_preserves_false_values(self) -> None:
        """All-false masks remain all-false after upscale."""
        masks = torch.zeros(2, 64, 64, dtype=torch.bool)
        result = upscale_masks(masks, orig_size=(256, 256))

        assert not result.any()

    def test_upscale_empty_masks(self) -> None:
        """Empty mask tensor is returned unchanged."""
        masks = torch.empty(0, 64, 64, dtype=torch.bool)
        result = upscale_masks(masks, orig_size=(256, 256))

        assert result.numel() == 0

    def test_upscale_single_mask(self) -> None:
        """Single mask is correctly upscaled."""
        masks = torch.zeros(1, 10, 10, dtype=torch.bool)
        masks[0, 3:7, 3:7] = True  # center square
        result = upscale_masks(masks, orig_size=(100, 100))

        assert result.shape == (1, 100, 100)
        assert result.dtype == torch.bool
        # Center region should be true, corners should be false
        assert result[0, 50, 50].item()
        assert not result[0, 0, 0].item()


class TestDownscaleResult:
    """Tests for the DownscaleResult dataclass."""

    def test_is_scaled_true_when_scale_below_one(self) -> None:
        """is_scaled returns True when scale < 1.0."""
        result = DownscaleResult(image=torch.empty(3, 100, 100), scale=0.5, orig_size=(200, 200))
        assert result.is_scaled

    def test_is_scaled_false_when_scale_is_one(self) -> None:
        """is_scaled returns False when scale == 1.0."""
        result = DownscaleResult(image=torch.empty(3, 200, 200), scale=1.0, orig_size=(200, 200))
        assert not result.is_scaled


class TestRoundTrip:
    """Tests for downscale + upscale round-trip consistency."""

    def test_round_trip_preserves_mask_coverage(self) -> None:
        """Downscaling an image then upscaling masks preserves rough spatial coverage."""
        h, w = 2776, 2082
        image = torch.randn(3, h, w)

        downscaled = downscale_image(image, max_side=1024)
        work_h, work_w = downscaled.image.shape[1], downscaled.image.shape[2]

        # Create masks at working resolution with a filled region
        masks = torch.zeros(1, work_h, work_w, dtype=torch.bool)
        masks[0, 10 : work_h - 10, 10 : work_w - 10] = True

        restored = upscale_masks(masks, orig_size=(h, w))

        assert restored.shape == (1, h, w)
        # Center should be true, edges should be false
        assert restored[0, h // 2, w // 2].item()
        assert not restored[0, 0, 0].item()
