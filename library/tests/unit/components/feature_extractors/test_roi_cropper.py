# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for ROICropper class."""

import pytest
import torch

from getiprompt.components.feature_extractors.roi_cropper import (
    CropRegion,
    CropResult,
    ROICropper,
    compute_centered_crop_region,
    compute_roi_crop_size,
    get_mask_bounding_box,
)


class TestGetMaskBoundingBox:
    """Test cases for get_mask_bounding_box function."""

    def test_simple_mask(self) -> None:
        """Test bounding box extraction from a simple mask."""
        mask = torch.zeros(100, 100)
        mask[20:40, 30:60] = 1  # Object from y=20-40, x=30-60

        x_min, y_min, x_max, y_max = get_mask_bounding_box(mask)

        pytest.assume(x_min == 30)
        pytest.assume(y_min == 20)
        pytest.assume(x_max == 60)
        pytest.assume(y_max == 40)

    def test_mask_with_channel_dim(self) -> None:
        """Test bounding box extraction from a 3D mask."""
        mask = torch.zeros(1, 100, 100)
        mask[0, 20:40, 30:60] = 1

        x_min, y_min, x_max, y_max = get_mask_bounding_box(mask)

        pytest.assume(x_min == 30)
        pytest.assume(y_min == 20)
        pytest.assume(x_max == 60)
        pytest.assume(y_max == 40)

    def test_empty_mask(self) -> None:
        """Test bounding box extraction from an empty mask."""
        mask = torch.zeros(100, 200)

        x_min, y_min, x_max, y_max = get_mask_bounding_box(mask)

        # Should return full image bounds for empty mask
        pytest.assume(x_min == 0)
        pytest.assume(y_min == 0)
        pytest.assume(x_max == 200)
        pytest.assume(y_max == 100)

    def test_single_pixel_mask(self) -> None:
        """Test bounding box extraction from a single pixel mask."""
        mask = torch.zeros(100, 100)
        mask[50, 75] = 1

        x_min, y_min, x_max, y_max = get_mask_bounding_box(mask)

        pytest.assume(x_min == 75)
        pytest.assume(y_min == 50)
        pytest.assume(x_max == 76)
        pytest.assume(y_max == 51)

    def test_full_mask(self) -> None:
        """Test bounding box extraction from a full mask."""
        mask = torch.ones(100, 200)

        x_min, y_min, x_max, y_max = get_mask_bounding_box(mask)

        pytest.assume(x_min == 0)
        pytest.assume(y_min == 0)
        pytest.assume(x_max == 200)
        pytest.assume(y_max == 100)


class TestComputeROICropSize:
    """Test cases for compute_roi_crop_size function."""

    def test_small_object(self) -> None:
        """Test crop size computation for a small object."""
        # Small object: 50x30 pixels
        crop_size = compute_roi_crop_size(
            bbox_width=50,
            bbox_height=30,
            input_size=512,
            patch_size=16,
            min_object_ratio=0.1,
            min_patches=2,
        )

        # Crop should be reasonable for preserving the object
        pytest.assume(crop_size >= 50)  # At least as large as object

    def test_large_object(self) -> None:
        """Test crop size computation for a large object."""
        # Large object: 500x400 pixels
        crop_size = compute_roi_crop_size(
            bbox_width=500,
            bbox_height=400,
            input_size=512,
            patch_size=16,
            min_object_ratio=0.1,
            min_patches=2,
        )

        pytest.assume(crop_size >= 500)  # At least as large as object

    def test_object_preservation(self) -> None:
        """Test that computed crop size preserves object in feature map."""
        bbox_width = 100
        bbox_height = 80
        input_size = 512
        patch_size = 16

        crop_size = compute_roi_crop_size(
            bbox_width=bbox_width,
            bbox_height=bbox_height,
            input_size=input_size,
            patch_size=patch_size,
            min_object_ratio=0.1,
            min_patches=2,
        )

        # Calculate object size in feature space after crop and resize
        feature_size = input_size // patch_size
        obj_feat_w = (bbox_width / crop_size) * feature_size
        obj_feat_h = (bbox_height / crop_size) * feature_size

        # Object should occupy at least 2 patches (min_patches)
        pytest.assume(min(obj_feat_w, obj_feat_h) >= 2)


class TestComputeCenteredCropRegion:
    """Test cases for compute_centered_crop_region function."""

    def test_centered_crop(self) -> None:
        """Test computing a centered crop region."""
        region = compute_centered_crop_region(
            x_min=100,
            y_min=100,
            x_max=200,
            y_max=200,
            crop_size=200,
            image_height=500,
            image_width=500,
        )

        # Crop should be centered on the bbox center (150, 150)
        crop_center_x = (region.x_min + region.x_max) // 2
        crop_center_y = (region.y_min + region.y_max) // 2

        # Center should be close to (150, 150)
        pytest.assume(abs(crop_center_x - 150) <= 1)
        pytest.assume(abs(crop_center_y - 150) <= 1)
        pytest.assume(region.width == 200)  # Correct width
        pytest.assume(region.height == 200)  # Correct height

    def test_crop_at_left_edge(self) -> None:
        """Test computing a crop region near the left edge."""
        region = compute_centered_crop_region(
            x_min=10,
            y_min=100,
            x_max=50,
            y_max=150,
            crop_size=200,
            image_height=500,
            image_width=500,
        )

        pytest.assume(region.x_min >= 0)
        pytest.assume(region.x_max <= 500)
        pytest.assume(region.width == 200)

    def test_crop_at_corner(self) -> None:
        """Test computing a crop region near a corner."""
        region = compute_centered_crop_region(
            x_min=0,
            y_min=0,
            x_max=20,
            y_max=20,
            crop_size=100,
            image_height=500,
            image_width=500,
        )

        pytest.assume(region.x_min >= 0)
        pytest.assume(region.y_min >= 0)
        pytest.assume(region.x_max <= 500)
        pytest.assume(region.y_max <= 500)

    def test_crop_larger_than_image(self) -> None:
        """Test crop region when crop size exceeds image dimensions."""
        region = compute_centered_crop_region(
            x_min=50,
            y_min=50,
            x_max=150,
            y_max=150,
            crop_size=400,
            image_height=200,
            image_width=200,
        )

        # Should be clamped to image boundaries
        pytest.assume(region.x_min >= 0)
        pytest.assume(region.y_min >= 0)
        pytest.assume(region.x_max <= 200)
        pytest.assume(region.y_max <= 200)


class TestCropRegion:
    """Test cases for CropRegion dataclass."""

    def test_properties(self) -> None:
        """Test CropRegion width and height properties."""
        region = CropRegion(
            x_min=10,
            y_min=20,
            x_max=110,
            y_max=120,
        )

        pytest.assume(region.width == 100)
        pytest.assume(region.height == 100)


class TestCropResult:
    """Test cases for CropResult dataclass."""

    def test_crop_result(self) -> None:
        """Test CropResult dataclass."""
        region = CropRegion(x_min=10, y_min=20, x_max=110, y_max=120)
        result = CropResult(
            cropped_image=torch.ones(3, 100, 100),
            cropped_mask=torch.ones(100, 100),
            crop_region=region,
        )

        pytest.assume(result.cropped_image.shape == (3, 100, 100))
        pytest.assume(result.cropped_mask.shape == (100, 100))
        pytest.assume(result.crop_region.width == 100)


class TestROICropper:
    """Test cases for ROICropper class."""

    def test_init(self) -> None:
        """Test ROICropper initialization."""
        cropper = ROICropper(
            input_size=512,
            patch_size=16,
            min_object_ratio=0.1,
            min_patches=2,
        )

        pytest.assume(cropper.input_size == 512)
        pytest.assume(cropper.patch_size == 16)
        pytest.assume(cropper.min_object_ratio == 0.1)
        pytest.assume(cropper.min_patches == 2)

    def test_should_crop_small_object(self) -> None:
        """Test should_crop returns True for small objects in large images."""
        cropper = ROICropper(input_size=512, patch_size=16, min_patches=2)

        # Very large image with tiny object
        mask = torch.zeros(10000, 10000)
        mask[5000:5010, 5000:5010] = 1  # 10x10 pixel object

        pytest.assume(cropper.should_crop(mask) is True)

    def test_should_crop_large_object(self) -> None:
        """Test should_crop returns False for large objects."""
        cropper = ROICropper(input_size=512, patch_size=16, min_patches=2)

        # Reasonable sized object relative to image
        mask = torch.zeros(512, 512)
        mask[100:300, 100:300] = 1  # 200x200 pixel object

        pytest.assume(cropper.should_crop(mask) is False)

    def test_compute_crop_region(self) -> None:
        """Test crop region computation."""
        cropper = ROICropper(input_size=512, patch_size=16)

        mask = torch.zeros(1000, 2000)
        mask[400:500, 900:1100] = 1  # 100x200 pixel object

        crop_region = cropper.compute_crop_region(mask)

        # Crop should contain the object
        pytest.assume(crop_region.x_min <= 900)
        pytest.assume(crop_region.x_max >= 1100)
        pytest.assume(crop_region.y_min <= 400)
        pytest.assume(crop_region.y_max >= 500)

    def test_crop_image_and_mask(self) -> None:
        """Test cropping both image and mask."""
        cropper = ROICropper(input_size=512, patch_size=16)

        # Create image and mask
        image = torch.rand(3, 1000, 2000)
        mask = torch.zeros(1000, 2000)
        mask[400:500, 900:1100] = 1

        crop_result = cropper.crop_image_and_mask(image, mask)

        # Check that cropped outputs have matching spatial dimensions
        pytest.assume(crop_result.cropped_image.shape[1] == crop_result.crop_region.height)
        pytest.assume(crop_result.cropped_image.shape[2] == crop_result.crop_region.width)
        pytest.assume(crop_result.cropped_mask.shape[0] == crop_result.crop_region.height)
        pytest.assume(crop_result.cropped_mask.shape[1] == crop_result.crop_region.width)

        # Check that mask is not empty after cropping
        pytest.assume(crop_result.cropped_mask.sum() > 0)

    def test_crop_image_and_mask_with_3d_mask(self) -> None:
        """Test cropping with 3D mask input."""
        cropper = ROICropper(input_size=512, patch_size=16)

        image = torch.rand(3, 1000, 1000)
        mask = torch.zeros(1, 1000, 1000)
        mask[0, 400:500, 400:500] = 1

        crop_result = cropper.crop_image_and_mask(image, mask)

        # Output mask should also be 3D
        pytest.assume(crop_result.cropped_mask.ndim == 3)
        pytest.assume(crop_result.cropped_mask.sum() > 0)

    def test_empty_mask(self) -> None:
        """Test crop region computation for an empty mask."""
        cropper = ROICropper(input_size=512, patch_size=16)

        mask = torch.zeros(1000, 2000)

        crop_region = cropper.compute_crop_region(mask)

        # Should return full image bounds for empty mask
        pytest.assume(crop_region.x_min == 0)
        pytest.assume(crop_region.y_min == 0)
        pytest.assume(crop_region.x_max == 2000)
        pytest.assume(crop_region.y_max == 1000)


class TestROICropperIntegration:
    """Integration tests for ROICropper with realistic scenarios."""

    def test_small_object_in_large_image(self) -> None:
        """Test the specific scenario from the issue: small object in large image.

        Original issue: mask size ~[6440, 12880] with ~51224 total pixels.
        Object disappears after resize to input_size and MaxPool2d.

        Solution: Crop BOTH image and mask around ROI before encoding.
        """
        cropper = ROICropper(
            input_size=512,
            patch_size=16,
            min_object_ratio=0.1,
            min_patches=2,
        )

        # Simulate the problematic case
        h, w = 6440, 12880

        # Create a small object (~51224 pixels)
        # sqrt(51224) ≈ 226, so roughly 226x226 object
        obj_h, obj_w = 226, 227
        y_start = h // 2 - obj_h // 2
        x_start = w // 2 - obj_w // 2

        # Create image and mask
        image = torch.rand(3, h, w)
        mask = torch.zeros(h, w)
        mask[y_start : y_start + obj_h, x_start : x_start + obj_w] = 1

        pytest.assume(mask.sum().item() == obj_h * obj_w)

        # Without cropping, check if object would survive
        feature_size = 512 // 16  # 32
        obj_feat_w = (obj_w / w) * feature_size  # ~0.56
        obj_feat_h = (obj_h / h) * feature_size  # ~1.12

        # Object would be less than 2 patches in feature space
        pytest.assume(min(obj_feat_w, obj_feat_h) < 2)

        # Verify cropping is needed
        pytest.assume(cropper.should_crop(mask, image_height=h, image_width=w) is True)

        # Crop both image and mask
        crop_result = cropper.crop_image_and_mask(image, mask)

        # Verify crop result is valid
        pytest.assume(crop_result.crop_region.width > 0)
        pytest.assume(crop_result.crop_region.height > 0)
        pytest.assume(crop_result.cropped_mask.sum() > 0)

        # Verify object now occupies more of the crop
        crop_h = crop_result.crop_region.height
        crop_w = crop_result.crop_region.width
        new_obj_feat_w = (obj_w / crop_w) * feature_size
        new_obj_feat_h = (obj_h / crop_h) * feature_size

        # After cropping, object should occupy >= 2 patches
        pytest.assume(min(new_obj_feat_w, new_obj_feat_h) >= 2)

        # Verify image and mask have matching crop dimensions
        pytest.assume(crop_result.cropped_image.shape[1] == crop_result.cropped_mask.shape[0])
        pytest.assume(crop_result.cropped_image.shape[2] == crop_result.cropped_mask.shape[1])

    def test_workflow_simulation(self) -> None:
        """Simulate the complete workflow: check -> crop -> encode (mock)."""
        cropper = ROICropper(input_size=512, patch_size=16)

        # Large image with small object
        image = torch.rand(3, 2000, 4000)
        mask = torch.zeros(2000, 4000)
        mask[900:1000, 1900:2100] = 1  # 100x200 pixel object

        # Step 1: Check if cropping is needed
        needs_crop = cropper.should_crop(mask)
        pytest.assume(needs_crop is True)

        # Step 2: Crop both image and mask
        crop_result = cropper.crop_image_and_mask(image, mask)

        # Step 3: Verify cropped image can be encoded (shape is reasonable)
        pytest.assume(crop_result.cropped_image.shape[0] == 3)  # 3 channels
        pytest.assume(crop_result.cropped_image.shape[1] > 0)  # Has height
        pytest.assume(crop_result.cropped_image.shape[2] > 0)  # Has width

        # Step 4: Verify cropped mask matches
        pytest.assume(crop_result.cropped_mask.sum() > 0)  # Mask has content
