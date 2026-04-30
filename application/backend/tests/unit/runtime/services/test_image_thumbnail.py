# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from runtime.services.image_thumbnail import generate_image_thumbnail


@pytest.fixture
def test_image(tmp_path):
    """Create a test image."""
    img_path = tmp_path / "test.jpg"
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, :] = [100, 150, 200]  # BGR color
    cv2.imwrite(str(img_path), img)
    return img_path


class TestGenerateImageThumbnail:
    def test_generate_thumbnail_valid_image(self, test_image):
        """Test thumbnail generation for a valid image."""
        thumbnail = generate_image_thumbnail(test_image)

        assert thumbnail is not None
        assert isinstance(thumbnail, str)
        assert thumbnail.startswith("data:image/jpeg;base64,")
        assert len(thumbnail) > 50  # Has actual content

    def test_generate_thumbnail_nonexistent_file(self):
        """Test thumbnail generation for non-existent file."""
        thumbnail = generate_image_thumbnail(Path("/nonexistent/file.jpg"))

        assert thumbnail is None

    def test_generate_thumbnail_invalid_image(self, tmp_path):
        """Test thumbnail generation for invalid image data."""
        invalid_path = tmp_path / "invalid.jpg"
        invalid_path.write_bytes(b"not an image")

        thumbnail = generate_image_thumbnail(invalid_path)

        assert thumbnail is None

    def test_generate_thumbnail_uses_settings_defaults(self, test_image, monkeypatch):
        """Test that settings defaults are used when no parameters provided."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.thumbnail_max_dimension = 300
        mock_settings.thumbnail_jpeg_quality = 85

        monkeypatch.setattr("runtime.services.image_thumbnail.settings", mock_settings)

        with patch("cv2.resize") as mock_resize, patch("cv2.imencode") as mock_encode:
            # Setup mocks
            mock_resize.return_value = np.zeros((150, 150, 3), dtype=np.uint8)
            mock_encode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))

            # Call with no parameters
            generate_image_thumbnail(test_image)

            # Verify imencode was called with settings.thumbnail_jpeg_quality
            mock_encode.assert_called_once()
            call_args = mock_encode.call_args[0]
            assert call_args[0] == ".jpg"
            assert call_args[2] == [cv2.IMWRITE_JPEG_QUALITY, 85]

    def test_generate_thumbnail_respects_custom_max_size(self, test_image):
        """Test that custom max_size is respected."""
        thumbnail_small = generate_image_thumbnail(test_image, max_size=50)
        thumbnail_large = generate_image_thumbnail(test_image, max_size=200)

        assert thumbnail_small is not None
        assert thumbnail_large is not None
        # Smaller max_size should result in smaller base64 string
        assert len(thumbnail_small) < len(thumbnail_large)

    def test_generate_thumbnail_respects_custom_jpeg_quality(self, test_image):
        """Test that custom jpeg_quality is respected."""
        thumbnail_low = generate_image_thumbnail(test_image, jpeg_quality=50)
        thumbnail_high = generate_image_thumbnail(test_image, jpeg_quality=95)

        assert thumbnail_low is not None
        assert thumbnail_high is not None
        # Higher quality should result in larger base64 string
        assert len(thumbnail_low) < len(thumbnail_high)

    def test_generate_thumbnail_custom_parameters_override_settings(self, test_image, monkeypatch):
        """Test that custom parameters override settings defaults."""
        # Mock settings with different values
        mock_settings = Mock()
        mock_settings.thumbnail_max_dimension = 300
        mock_settings.thumbnail_jpeg_quality = 85

        monkeypatch.setattr("runtime.services.image_thumbnail.settings", mock_settings)

        with patch("cv2.resize") as mock_resize, patch("cv2.imencode") as mock_encode:
            # Setup mocks
            mock_resize.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_encode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))

            # Call with custom parameters
            generate_image_thumbnail(test_image, max_size=100, jpeg_quality=60)

            # Verify imencode was called with custom jpeg_quality, not settings default
            mock_encode.assert_called_once()
            call_args = mock_encode.call_args[0]
            assert call_args[2] == [cv2.IMWRITE_JPEG_QUALITY, 60]

    def test_generate_thumbnail_resizes_large_image(self, tmp_path):
        """Test that large images are resized according to max_size."""
        large_img_path = tmp_path / "large.jpg"
        large_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        cv2.imwrite(str(large_img_path), large_img)

        thumbnail = generate_image_thumbnail(large_img_path, max_size=200)

        assert thumbnail is not None
        # Decode and verify size
        import base64

        img_data = base64.b64decode(thumbnail.split(",")[1])
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        decoded_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Should be resized to max_size
        assert max(decoded_img.shape[:2]) <= 200

    def test_generate_thumbnail_preserves_aspect_ratio(self, tmp_path):
        """Test that aspect ratio is preserved when resizing."""
        rect_img_path = tmp_path / "rectangular.jpg"
        # Create a 400x200 image (2:1 aspect ratio)
        rect_img = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(rect_img_path), rect_img)

        thumbnail = generate_image_thumbnail(rect_img_path, max_size=100)

        assert thumbnail is not None
        # Decode and check aspect ratio
        import base64

        img_data = base64.b64decode(thumbnail.split(",")[1])
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        decoded_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        height, width = decoded_img.shape[:2]
        aspect_ratio = width / height
        # Should maintain approximately 2:1 aspect ratio
        assert 1.9 < aspect_ratio < 2.1

    def test_generate_thumbnail_no_upscaling_small_image(self, tmp_path):
        """Test that small images are not upscaled beyond their original size."""
        small_img_path = tmp_path / "small.jpg"
        # Create a 50x50 image (smaller than max_size default of 300)
        small_img = np.zeros((50, 50, 3), dtype=np.uint8)
        small_img[:, :] = [100, 150, 200]
        cv2.imwrite(str(small_img_path), small_img)

        thumbnail = generate_image_thumbnail(small_img_path, max_size=300)

        assert thumbnail is not None
        # Decode and verify it wasn't upscaled
        import base64

        img_data = base64.b64decode(thumbnail.split(",")[1])
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        decoded_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        height, width = decoded_img.shape[:2]
        # Should remain at original size, not upscaled to 300x300
        assert height <= 50
        assert width <= 50

    def test_generate_thumbnail_very_thin_image(self, tmp_path):
        """Test that very thin images don't cause errors due to 0-dimension rounding."""
        thin_img_path = tmp_path / "thin.jpg"
        # Create a 1x1000 image (very thin)
        thin_img = np.zeros((1000, 1, 3), dtype=np.uint8)
        thin_img[:, :] = [100, 150, 200]
        cv2.imwrite(str(thin_img_path), thin_img)

        # This should not raise an error even with aggressive downscaling
        thumbnail = generate_image_thumbnail(thin_img_path, max_size=100)

        assert thumbnail is not None
        # Decode and verify dimensions are at least 1px
        import base64

        img_data = base64.b64decode(thumbnail.split(",")[1])
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        decoded_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        height, width = decoded_img.shape[:2]
        # Both dimensions should be at least 1
        assert height >= 1
        assert width >= 1
        # Longest dimension should be capped at max_size
        assert max(height, width) <= 100

    def test_generate_thumbnail_tiny_image(self, tmp_path):
        """Test that tiny images (smaller than max_size) are not upscaled."""
        tiny_img_path = tmp_path / "tiny.jpg"
        # Create a 10x20 image
        tiny_img = np.zeros((20, 10, 3), dtype=np.uint8)
        tiny_img[:, :] = [100, 150, 200]
        cv2.imwrite(str(tiny_img_path), tiny_img)

        thumbnail = generate_image_thumbnail(tiny_img_path, max_size=300)

        assert thumbnail is not None
        # Decode and verify original size is preserved
        import base64

        img_data = base64.b64decode(thumbnail.split(",")[1])
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        decoded_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        height, width = decoded_img.shape[:2]
        # Should maintain original dimensions
        assert height == 20
        assert width == 10
