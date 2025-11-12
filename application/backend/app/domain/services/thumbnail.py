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

from domain.errors import ServiceError
from domain.services.schemas.annotation import (
    AnnotationSchema,
    AnnotationType,
    PolygonAnnotation,
    RectangleAnnotation,
)
from domain.services.schemas.label import LabelSchema
from settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


def generate_thumbnail(frame: np.ndarray, annotations: list[tuple[AnnotationSchema, LabelSchema]]) -> str:
    """
    Generate a thumbnail with annotation overlays.

    Args:
        frame: Original frame as numpy array
        annotations: List of (annotation, label) tuples

    Returns:
        Base64-encoded data URI of the thumbnail with JPEG encoding
    """
    try:
        thumbnail = _resize_frame_to_thumbnail_size(frame)
        height, width = thumbnail.shape[:2]
        line_thickness = max(
            settings.thumbnail_min_line_thickness,
            int(min(height, width) * settings.thumbnail_line_thickness_ratio),
        )

        # create overlay for semi-transparent annotations
        annotation_overlay = thumbnail.copy()

        for annotation_schema, label in annotations:
            color_bgr = _convert_hex_to_bgr(label.color)
            annotation = annotation_schema.config

            if annotation.type == AnnotationType.RECTANGLE:
                annotation_overlay = _draw_filled_rectangle(
                    annotation_overlay, annotation, color_bgr, width, height, line_thickness
                )
            elif annotation.type == AnnotationType.POLYGON:
                annotation_overlay = _draw_filled_polygon(
                    annotation_overlay, annotation, color_bgr, width, height, line_thickness
                )
            else:
                logger.warning(f"Unsupported annotation type: {annotation.type}")

        # blend annotation overlay with original thumbnail for semi-transparency
        cv2.addWeighted(
            annotation_overlay,
            settings.thumbnail_fill_opacity,
            thumbnail,
            1 - settings.thumbnail_fill_opacity,
            0,
            thumbnail,
        )

        return _encode_image_to_base64_data_uri(thumbnail)
    except (cv2.error, ValueError) as e:
        raise ServiceError(f"Failed to generate thumbnail: {str(e)}")


def _resize_frame_to_thumbnail_size(frame: np.ndarray) -> np.ndarray:
    """
    Resize frame to thumbnail dimensions while maintaining aspect ratio.

    Args:
        frame: Original frame as numpy array

    Returns:
        Resized frame, or copy of original if already small enough
    """
    height, width = frame.shape[:2]

    if max(height, width) <= settings.thumbnail_max_dimension:
        return frame.copy()

    scale_factor = settings.thumbnail_max_dimension / max(height, width)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _draw_filled_rectangle(
    overlay: np.ndarray,
    rect: RectangleAnnotation,
    color_bgr: tuple[int, int, int],
    image_width: int,
    image_height: int,
    border_thickness: int,
) -> np.ndarray:
    """
    Draw a rectangle annotation with semi-transparent fill and opaque border.

    Args:
        overlay: Image overlay to draw on
        rect: Rectangle annotation with normalized coordinates
        color_bgr: Color in BGR format
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        border_thickness: Thickness of the rectangle border

    Returns:
        Overlay with rectangle drawn
    """
    top_left = (int(rect.points[0].x * image_width), int(rect.points[0].y * image_height))
    bottom_right = (int(rect.points[1].x * image_width), int(rect.points[1].y * image_height))

    # draw filled rectangle
    cv2.rectangle(overlay, top_left, bottom_right, color_bgr, -1)

    # draw border with full opacity for better visibility
    cv2.rectangle(overlay, top_left, bottom_right, color_bgr, border_thickness)

    return overlay


def _draw_filled_polygon(
    overlay: np.ndarray,
    polygon: PolygonAnnotation,
    color_bgr: tuple[int, int, int],
    image_width: int,
    image_height: int,
    border_thickness: int,
) -> np.ndarray:
    """
    Draw a polygon annotation with semi-transparent fill and opaque border.

    Args:
        overlay: Image overlay to draw on
        polygon: Polygon annotation with normalized coordinates
        color_bgr: Color in BGR format
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        border_thickness: Thickness of the polygon border

    Returns:
        Overlay with polygon drawn
    """
    pixel_points = np.array(
        [[int(pt.x * image_width), int(pt.y * image_height)] for pt in polygon.points],
        dtype=np.int32,
    )

    # draw filled polygon
    cv2.fillPoly(overlay, [pixel_points], color_bgr)

    # draw border with full opacity for better visibility
    cv2.polylines(overlay, [pixel_points], isClosed=True, color=color_bgr, thickness=border_thickness)

    return overlay


def _convert_hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """
    Convert hex color string to BGR tuple for OpenCV.

    Args:
        hex_color: Hex color string (e.g., "#FF0000" or "FF0000")

    Returns:
        BGR color tuple (e.g., (0, 0, 255) for red)
    """
    hex_color = hex_color.lstrip("#")
    red, green, blue = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return blue, green, red  # OpenCV uses BGR instead of RGB


def _encode_image_to_base64_data_uri(image: np.ndarray) -> str:
    """
    Encode image to base64 data URI with JPEG format.

    Args:
        image: Image as numpy array

    Returns:
        Data URI string (e.g., "data:image/jpeg;base64,...")

    Raises:
        RuntimeError: If encoding fails
    """
    encode_success, encoded_buffer = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, settings.thumbnail_jpeg_quality]
    )
    if not encode_success:
        raise RuntimeError("Failed to encode thumbnail to JPEG format")

    base64_bytes = base64.b64encode(encoded_buffer.tobytes())
    base64_string = base64_bytes.decode("utf-8")

    return f"data:image/jpeg;base64,{base64_string}"
