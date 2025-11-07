# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np

from domain.services.schemas.annotation import (
    AnnotationSchema,
    Point,
    PolygonAnnotation,
    RectangleAnnotation,
)
from domain.services.thumbnail import (
    THUMBNAIL_MAX_DIMENSION,
    _convert_hex_to_bgr,
    _draw_filled_polygon,
    _draw_filled_rectangle,
    _encode_image_to_base64_data_uri,
    _resize_frame_to_thumbnail_size,
    generate_thumbnail,
)


def make_label(color="#FF0000"):
    return SimpleNamespace(id=uuid4(), name="test", color=color)


def create_test_frame(width=800, height=600, color=(100, 150, 200)):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = color
    return frame


class TestResizeFrame:
    def test_resize_large_frame(self):
        frame = create_test_frame(width=1000, height=800)
        resized = _resize_frame_to_thumbnail_size(frame)

        assert max(resized.shape[:2]) == THUMBNAIL_MAX_DIMENSION
        assert resized.shape[0] == 240  # 800 * (300/1000)
        assert resized.shape[1] == 300

    def test_resize_tall_frame(self):
        frame = create_test_frame(width=600, height=1200)
        resized = _resize_frame_to_thumbnail_size(frame)

        assert max(resized.shape[:2]) == THUMBNAIL_MAX_DIMENSION
        assert resized.shape[0] == 300
        assert resized.shape[1] == 150  # 600 * (300/1200)

    def test_no_resize_small_frame(self):
        frame = create_test_frame(width=300, height=200)
        resized = _resize_frame_to_thumbnail_size(frame)

        assert resized.shape == frame.shape
        np.testing.assert_array_equal(resized, frame)

    def test_aspect_ratio_preserved(self):
        frame = create_test_frame(width=1600, height=900)
        resized = _resize_frame_to_thumbnail_size(frame)

        original_ratio = 1600 / 900
        resized_ratio = resized.shape[1] / resized.shape[0]

        np.testing.assert_almost_equal(original_ratio, resized_ratio, decimal=2)


class TestHexToBgr:
    def test_hex_with_hash(self):
        assert _convert_hex_to_bgr("#FF0000") == (0, 0, 255)  # Red in BGR
        assert _convert_hex_to_bgr("#00FF00") == (0, 255, 0)  # Green in BGR
        assert _convert_hex_to_bgr("#0000FF") == (255, 0, 0)  # Blue in BGR

    def test_hex_without_hash(self):
        assert _convert_hex_to_bgr("FF0000") == (0, 0, 255)
        assert _convert_hex_to_bgr("00FF00") == (0, 255, 0)

    def test_various_colors(self):
        assert _convert_hex_to_bgr("#FFFFFF") == (255, 255, 255)  # White
        assert _convert_hex_to_bgr("#000000") == (0, 0, 0)  # Black
        assert _convert_hex_to_bgr("#FF8800") == (0, 136, 255)  # Orange


class TestEncodeToBase64:
    def test_encode_valid_image(self):
        frame = create_test_frame(width=100, height=100)
        result = _encode_image_to_base64_data_uri(frame)

        assert result.startswith("data:image/jpeg;base64,")
        assert len(result) > 50

        # verify it's valid base64
        b64_data = result.split(",")[1]
        decoded = base64.b64decode(b64_data)
        assert len(decoded) > 0

    def test_encoded_image_decodable(self):
        frame = create_test_frame(width=200, height=150, color=(50, 100, 150))
        encoded = _encode_image_to_base64_data_uri(frame)

        b64_data = encoded.split(",")[1]
        decoded_bytes = base64.b64decode(b64_data)
        decoded_img = cv2.imdecode(np.frombuffer(decoded_bytes, np.uint8), cv2.IMREAD_COLOR)

        assert decoded_img is not None
        assert decoded_img.shape == frame.shape


class TestDrawRectangle:
    def test_draw_rectangle_basic(self):
        overlay = create_test_frame(width=200, height=200, color=(255, 255, 255))
        rect = RectangleAnnotation(type="rectangle", points=[Point(x=0.2, y=0.2), Point(x=0.8, y=0.8)])
        color = (0, 0, 255)

        _draw_filled_rectangle(overlay, rect, color, image_width=200, image_height=200, border_thickness=2)

        # check that some pixels in the rectangle area have been modified
        center_pixel = overlay[100, 100]
        np.testing.assert_array_equal(center_pixel, color)

    def test_draw_rectangle_coordinates(self):
        overlay = np.zeros((100, 100, 3), dtype=np.uint8)
        rect = RectangleAnnotation(type="rectangle", points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.5)])
        color = (255, 0, 0)

        _draw_filled_rectangle(overlay, rect, color, image_width=100, image_height=100, border_thickness=1)

        # check corners are colored
        assert np.any(overlay[10, 10] == color)
        assert np.any(overlay[50, 50] == color)


class TestDrawPolygon:
    def test_draw_triangle(self):
        overlay = create_test_frame(width=200, height=200, color=(255, 255, 255))
        polygon = PolygonAnnotation(
            type="polygon", points=[Point(x=0.5, y=0.1), Point(x=0.1, y=0.9), Point(x=0.9, y=0.9)]
        )
        color = (0, 255, 0)

        _draw_filled_polygon(overlay, polygon, color, image_width=200, image_height=200, border_thickness=2)

        # check that center of triangle has been colored
        center_pixel = overlay[120, 100]
        np.testing.assert_array_equal(center_pixel, color)

    def test_draw_square_polygon(self):
        overlay = np.zeros((100, 100, 3), dtype=np.uint8)
        polygon = PolygonAnnotation(
            type="polygon",
            points=[Point(x=0.2, y=0.2), Point(x=0.8, y=0.2), Point(x=0.8, y=0.8), Point(x=0.2, y=0.8)],
        )
        color = (0, 0, 255)

        _draw_filled_polygon(overlay, polygon, color, image_width=100, image_height=100, border_thickness=1)

        # check center is colored
        assert np.any(overlay[50, 50] == color)


class TestGenerateThumbnail:
    def test_generate_thumbnail_with_rectangle(self):
        frame = create_test_frame(width=800, height=600)
        label_id = uuid4()
        annotation = AnnotationSchema(
            config=RectangleAnnotation(type="rectangle", points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.5)]),
            label_id=label_id,
        )
        label = make_label(color="#FF0000")

        result = generate_thumbnail(frame, [(annotation, label)])

        assert result.startswith("data:image/jpeg;base64,")
        assert len(result) > 100

    def test_generate_thumbnail_with_polygon(self):
        frame = create_test_frame(width=600, height=400)
        label_id = uuid4()
        annotation = AnnotationSchema(
            config=PolygonAnnotation(
                type="polygon",
                points=[Point(x=0.2, y=0.2), Point(x=0.8, y=0.2), Point(x=0.8, y=0.8), Point(x=0.2, y=0.8)],
            ),
            label_id=label_id,
        )
        label = make_label(color="#00FF00")

        result = generate_thumbnail(frame, [(annotation, label)])

        assert result.startswith("data:image/jpeg;base64,")

    def test_generate_thumbnail_multiple_annotations(self):
        frame = create_test_frame(width=1000, height=800)
        label_id_1 = uuid4()
        label_id_2 = uuid4()
        annotations = [
            (
                AnnotationSchema(
                    config=RectangleAnnotation(type="rectangle", points=[Point(x=0.1, y=0.1), Point(x=0.3, y=0.3)]),
                    label_id=label_id_1,
                ),
                make_label(color="#FF0000"),
            ),
            (
                AnnotationSchema(
                    config=PolygonAnnotation(
                        type="polygon",
                        points=[Point(x=0.5, y=0.5), Point(x=0.9, y=0.5), Point(x=0.7, y=0.9)],
                    ),
                    label_id=label_id_2,
                ),
                make_label(color="#0000FF"),
            ),
        ]

        result = generate_thumbnail(frame, annotations)

        assert result.startswith("data:image/jpeg;base64,")

    def test_generate_thumbnail_resizes_large_frame(self):
        large_frame = create_test_frame(width=2000, height=1500)
        label_id = uuid4()
        annotation = AnnotationSchema(
            config=RectangleAnnotation(type="rectangle", points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.5)]),
            label_id=label_id,
        )
        label = make_label(color="#00FF00")

        result = generate_thumbnail(large_frame, [(annotation, label)])

        # decode to verify size
        b64_data = result.split(",")[1]
        decoded_bytes = base64.b64decode(b64_data)
        decoded_img = cv2.imdecode(np.frombuffer(decoded_bytes, np.uint8), cv2.IMREAD_COLOR)

        assert max(decoded_img.shape[:2]) <= THUMBNAIL_MAX_DIMENSION

    def test_generate_thumbnail_empty_annotations(self):
        frame = create_test_frame(width=400, height=300)

        result = generate_thumbnail(frame, [])

        assert result.startswith("data:image/jpeg;base64,")

    def test_generate_thumbnail_different_colors(self):
        frame = create_test_frame(width=500, height=500)
        colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"]

        for color in colors:
            label_id = uuid4()
            annotation = AnnotationSchema(
                config=RectangleAnnotation(type="rectangle", points=[Point(x=0.1, y=0.1), Point(x=0.3, y=0.3)]),
                label_id=label_id,
            )
            label = make_label(color=color)

            result = generate_thumbnail(frame, [(annotation, label)])

            assert result.startswith("data:image/jpeg;base64,")
            assert len(result) > 100
