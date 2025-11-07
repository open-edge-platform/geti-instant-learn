# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Thumbnail generation utilities for creating annotated frame previews.

This module provides pure functions for generating thumbnails with overlayed annotations.
Thumbnails are reduced-resolution images with semi-transparent annotations drawn using label colors.
"""

import base64
import logging

import cv2
import numpy as np

from domain.db.models import LabelDB
from domain.services.schemas.annotation import (
    AnnotationSchema,
    AnnotationType,
    PolygonAnnotation,
    RectangleAnnotation,
)

logger = logging.getLogger(__name__)

THUMBNAIL_MAX_DIMENSION = 400
LINE_THICKNESS_RATIO = 0.005  # 0.3% of image dimension
MIN_LINE_THICKNESS = 2
FILL_ALPHA = 0.3


def generate_thumbnail(frame: np.ndarray, annotations: list[tuple[AnnotationSchema, LabelDB]]) -> str:
    """
    Generate a thumbnail with annotation overlays.

    Args:
        frame: Original frame as numpy array
        annotations: List of (annotation, label) tuples

    Returns:
        Base64-encoded data URI of the thumbnail
    """
    thumbnail = _resize_frame(frame)
    height, width = thumbnail.shape[:2]
    line_thickness = max(
        MIN_LINE_THICKNESS,
        int(min(height, width) * LINE_THICKNESS_RATIO),
    )

    overlay = thumbnail.copy()

    for annotation_schema, label in annotations:
        color_bgr = _hex_to_bgr(label.color)
        annotation = annotation_schema.config

        if annotation.type == AnnotationType.RECTANGLE:
            _draw_rectangle(overlay, annotation, color_bgr, width, height, line_thickness)
        elif annotation.type == AnnotationType.POLYGON:
            _draw_polygon(overlay, annotation, color_bgr, width, height, line_thickness)

    # blend overlay with original for transparency
    cv2.addWeighted(
        overlay,
        FILL_ALPHA,
        thumbnail,
        1 - FILL_ALPHA,
        0,
        thumbnail,
    )

    return _encode_to_base64(thumbnail)


def _resize_frame(frame: np.ndarray) -> np.ndarray:
    """Resize frame maintaining aspect ratio."""
    height, width = frame.shape[:2]

    if max(height, width) <= THUMBNAIL_MAX_DIMENSION:
        return frame.copy()

    scale = THUMBNAIL_MAX_DIMENSION / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _draw_rectangle(
    overlay: np.ndarray,
    rect: RectangleAnnotation,
    color: tuple[int, int, int],
    width: int,
    height: int,
    thickness: int,
) -> None:
    """Draw a rectangle annotation with fill and border."""
    pt1 = (int(rect.points[0].x * width), int(rect.points[0].y * height))
    pt2 = (int(rect.points[1].x * width), int(rect.points[1].y * height))

    # draw filled rectangle on overlay
    cv2.rectangle(overlay, pt1, pt2, color, -1)

    # draw border on overlay with higher opacity
    cv2.rectangle(overlay, pt1, pt2, color, thickness)


def _draw_polygon(
    overlay: np.ndarray,
    polygon: PolygonAnnotation,
    color: tuple[int, int, int],
    width: int,
    height: int,
    thickness: int,
) -> None:
    """Draw a polygon annotation with fill and border."""
    points = np.array([[int(pt.x * width), int(pt.y * height)] for pt in polygon.points], dtype=np.int32)

    # draw filled polygon on overlay
    cv2.fillPoly(overlay, [points], color)

    # draw border on overlay with higher opacity
    cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=thickness)


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return rgb[2], rgb[1], rgb[0]  # convert RGB to BGR


def _encode_to_base64(image: np.ndarray) -> str:
    """Encode image to base64 data URI."""
    success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise RuntimeError("Failed to encode thumbnail")

    b64_bytes = base64.b64encode(buffer.tobytes())
    b64_str = b64_bytes.decode("utf-8")

    return f"data:image/jpeg;base64,{b64_str}"
