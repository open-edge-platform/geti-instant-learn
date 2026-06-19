# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import uuid

from domain.services.schemas.annotation import (
    AnnotationSchema,
    Point,
    PolygonAnnotation,
)
from domain.services.schemas.mappers.prompt import deduplicate_annotations


class TestDeduplicateAnnotations:
    def test_removes_exact_duplicates(self) -> None:
        label_id = uuid.uuid4()

        config1 = PolygonAnnotation(
            points=[Point(x=64, y=48), Point(x=320, y=48), Point(x=320, y=240), Point(x=64, y=240)],
        )
        config2 = PolygonAnnotation(
            points=[Point(x=64, y=48), Point(x=320, y=48), Point(x=320, y=240), Point(x=64, y=240)],
        )

        annotations = [
            AnnotationSchema(config=config1, label_id=label_id),
            AnnotationSchema(config=config2, label_id=label_id),
        ]

        result = deduplicate_annotations(annotations, 480, 640)

        assert len(result) == 1
        assert result[0].label_id == label_id

    def test_removes_similar_polygons(self) -> None:
        label_id = uuid.uuid4()

        config1 = PolygonAnnotation(
            points=[Point(x=64, y=48), Point(x=320, y=48), Point(x=320, y=240), Point(x=64, y=240)],
        )
        config2 = PolygonAnnotation(
            points=[Point(x=70, y=53), Point(x=326, y=53), Point(x=326, y=246), Point(x=70, y=246)],
        )

        annotations = [
            AnnotationSchema(config=config1, label_id=label_id),
            AnnotationSchema(config=config2, label_id=label_id),
        ]

        result = deduplicate_annotations(annotations, 480, 640, iou_threshold=0.9)

        assert len(result) == 1

    def test_keeps_different_polygons(self) -> None:
        label_id = uuid.uuid4()

        config1 = PolygonAnnotation(
            points=[Point(x=64, y=48), Point(x=192, y=48), Point(x=192, y=192), Point(x=64, y=192)],
        )
        config2 = PolygonAnnotation(
            points=[Point(x=384, y=240), Point(x=512, y=240), Point(x=512, y=384), Point(x=384, y=384)],
        )

        annotations = [
            AnnotationSchema(config=config1, label_id=label_id),
            AnnotationSchema(config=config2, label_id=label_id),
        ]

        result = deduplicate_annotations(annotations, 480, 640)

        assert len(result) == 2

    def test_keeps_non_duplicate_polygons(self) -> None:
        label_id = uuid.uuid4()

        polygon_a = PolygonAnnotation(
            points=[Point(x=64, y=48), Point(x=320, y=48), Point(x=320, y=240), Point(x=64, y=240)],
        )
        polygon_b = PolygonAnnotation(
            points=[Point(x=10, y=10), Point(x=50, y=10), Point(x=50, y=50), Point(x=10, y=50)],
        )

        annotations = [
            AnnotationSchema(config=polygon_a, label_id=label_id),
            AnnotationSchema(config=polygon_b, label_id=label_id),
        ]

        result = deduplicate_annotations(annotations, 480, 640)

        assert len(result) == 2
        assert all(isinstance(ann.config, PolygonAnnotation) for ann in result)

    def test_empty_list(self) -> None:
        result = deduplicate_annotations([], 480, 640)
        assert result == []

    def test_single_annotation(self) -> None:
        label_id = uuid.uuid4()
        config = PolygonAnnotation(
            points=[Point(x=64, y=48), Point(x=320, y=48), Point(x=320, y=240), Point(x=64, y=240)],
        )
        annotations = [AnnotationSchema(config=config, label_id=label_id)]

        result = deduplicate_annotations(annotations, 480, 640)

        assert len(result) == 1
        assert result[0].label_id == label_id
