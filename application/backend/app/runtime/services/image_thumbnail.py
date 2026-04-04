# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def generate_image_thumbnail(image_path: Path, max_size: int = 150, jpeg_quality: int = 80) -> str | None:
    """Generate a base64-encoded JPEG thumbnail for an image file."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        height, width = image.shape[:2]
        scale = min(max_size / width, max_size / height)
        new_width, new_height = int(width * scale), int(height * scale)
        thumbnail = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        success, buffer = cv2.imencode(".jpg", thumbnail, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not success:
            return None

        base64_string = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_string}"
    except Exception:
        logger.warning("Failed to generate thumbnail for %s", image_path, exc_info=True)
        return None
