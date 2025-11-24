# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from domain.services.schemas.annotation import AnnotationType, Point, PolygonAnnotation
from domain.services.schemas.mappers.mask import polygons_to_masks


class TestPolygonsToMasks:
    def test_polygons_converted_to_masks(self):
        polygon1 = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[
                Point(x=0.0, y=0.0),
                Point(x=0.3, y=0.0),
                Point(x=0.3, y=0.3),
                Point(x=0.0, y=0.3),
            ],
        )
        polygon2 = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[
                Point(x=0.7, y=0.7),
                Point(x=1.0, y=0.7),
                Point(x=1.0, y=1.0),
                Point(x=0.7, y=1.0),
            ],
        )

        masks = polygons_to_masks([polygon1, polygon2], image_height=100, image_width=100)

        assert masks.shape == (2, 100, 100)
        assert masks.dtype == np.uint8
        assert np.sum(masks[0]) > 0
        assert np.sum(masks[1]) > 0

    def test_empty_polygons_raises_exception(self):
        with pytest.raises(RuntimeError, match="Cannot convert empty polygons list to masks"):
            polygons_to_masks([], image_height=100, image_width=100)
