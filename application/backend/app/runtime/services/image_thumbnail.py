# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
from pathlib import Path

import cv2

from settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


def generate_image_thumbnail(
    image_path: Path,
    max_size: int | None = None,
    jpeg_quality: int | None = None,
) -> str | None:
    """Generate a base64-encoded JPEG thumbnail for an image file.

    Args:
        image_path: Path to the image file
        max_size: Maximum dimension for thumbnail. If None, use the configured setting.
        jpeg_quality: JPEG quality 0-100. If None, use the configured setting.

    Returns:
        Base64-encoded data URI string or None if generation fails
    """
    if max_size is None:
        max_size = settings.thumbnail_max_dimension
    if jpeg_quality is None:
        jpeg_quality = settings.thumbnail_jpeg_quality
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        height, width = image.shape[:2]
        # Cap scale at 1.0 to prevent upscaling small images
        scale = min(1.0, max_size / width, max_size / height)
        # Ensure dimensions are at least 1px to avoid cv2.resize errors
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        # Only resize if needed (scale < 1.0)
        if scale < 1.0:
            thumbnail = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            thumbnail = image

        success, buffer = cv2.imencode(".jpg", thumbnail, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not success:
            return None

        base64_string = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_string}"
    except Exception:
        logger.warning("Failed to generate thumbnail for %s", image_path, exc_info=True)
        return None
