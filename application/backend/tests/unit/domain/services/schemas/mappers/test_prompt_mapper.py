# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import uuid
from types import SimpleNamespace

import numpy as np
import pytest
from instantlearn.data.base.sample import Sample

from domain.db.models import PromptType
from domain.errors import ServiceError
from domain.services.schemas.annotation import (
    AnnotationSchema,
    AnnotationType,
    Point,
    PolygonAnnotation,
    RectangleAnnotation,
)
from domain.services.schemas.mappers.prompt import deduplicate_annotations, visual_prompt_to_sample
from domain.services.schemas.prompt import VisualPromptWithFrameSchema


class TestPromptMapper:
    @pytest.fixture
    def sample_frame(self) -> np.ndarray:
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_visual_prompt_to_sample_with_frame(self, sample_frame: np.ndarray) -> None:
        prompt_id = uuid.uuid4()
        frame_id = uuid.uuid4()
        label_id = uuid.uuid4()

        config = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.1), Point(x=0.5, y=0.5), Point(x=0.1, y=0.5)],
        )

        annotation = AnnotationSchema(config=config, label_id=label_id)

        prompt = VisualPromptWithFrameSchema(
            id=prompt_id,
            type=PromptType.VISUAL,
            frame_id=frame_id,
            annotations=[annotation],
            frame=sample_frame,
        )

        label_to_category_id = {label_id: 0}

        result = visual_prompt_to_sample(prompt, label_to_category_id)

        assert result is not None
        assert isinstance(result, Sample)
        assert np.array_equal(result.image.permute(1, 2, 0).numpy(), sample_frame)
        assert len(result.categories) == 1
        assert result.categories[0] == str(label_id)

    def test_visual_prompt_to_sample_raises_error_without_polygons(self, sample_frame: np.ndarray) -> None:
        prompt_id = uuid.uuid4()
        frame_id = uuid.uuid4()
        label_id = uuid.uuid4()

        config = RectangleAnnotation(type=AnnotationType.RECTANGLE, points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.5)])

        annotation = AnnotationSchema(config=config, label_id=label_id)

        prompt = VisualPromptWithFrameSchema(
            id=prompt_id,
            type=PromptType.VISUAL,
            frame_id=frame_id,
            annotations=[annotation],
            frame=sample_frame,
        )

        label_to_category_id = {label_id: 0}

        with pytest.raises(ServiceError, match="must have at least one polygon annotation"):
            visual_prompt_to_sample(prompt, label_to_category_id)

    def test_visual_prompt_to_sample_with_multiple_polygons(self, sample_frame: np.ndarray) -> None:
        prompt_id = uuid.uuid4()
        frame_id = uuid.uuid4()
        label_id_1 = uuid.uuid4()
        label_id_2 = uuid.uuid4()

        config_1 = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=0.1, y=0.1), Point(x=0.3, y=0.1), Point(x=0.3, y=0.3), Point(x=0.1, y=0.3)],
        )
        config_2 = PolygonAnnotation(
            type=AnnotationType.POLYGON, points=[Point(x=0.5, y=0.5), Point(x=0.7, y=0.7), Point(x=0.6, y=0.8)]
        )

        annotation_1 = AnnotationSchema(config=config_1, label_id=label_id_1)
        annotation_2 = AnnotationSchema(config=config_2, label_id=label_id_2)

        prompt = VisualPromptWithFrameSchema(
            id=prompt_id,
            type=PromptType.VISUAL,
            frame_id=frame_id,
            annotations=[annotation_1, annotation_2],
            frame=sample_frame,
        )

        label_to_category_id = {label_id_1: 0, label_id_2: 1}

        result = visual_prompt_to_sample(prompt, label_to_category_id)

        assert result is not None
        assert isinstance(result, Sample)
        assert len(result.categories) == 2
        assert str(label_id_1) in result.categories
        assert str(label_id_2) in result.categories

    def test_visual_prompt_to_sample_raises_error_for_text_prompt(self, sample_frame: np.ndarray) -> None:
        prompt_id = uuid.uuid4()
        frame_id = uuid.uuid4()

        # Using SimpleNamespace to bypass type restriction for testing error case
        prompt = SimpleNamespace(
            id=prompt_id,
            type=PromptType.TEXT,
            frame_id=frame_id,
            annotations=[],
            frame=sample_frame,
        )

        label_to_category_id: dict[uuid.UUID, int] = {}

        with pytest.raises(ServiceError, match="Cannot convert non-visual prompt"):
            visual_prompt_to_sample(prompt, label_to_category_id)  # type: ignore[arg-type]

    def test_visual_prompt_to_sample_raises_error_without_annotations(self, sample_frame: np.ndarray) -> None:
        prompt_id = uuid.uuid4()
        frame_id = uuid.uuid4()

        prompt = VisualPromptWithFrameSchema(
            id=prompt_id,
            type=PromptType.VISUAL,
            frame_id=frame_id,
            annotations=[],
            frame=sample_frame,
        )

        label_to_category_id: dict[uuid.UUID, int] = {}

        with pytest.raises(ServiceError, match="has no valid annotations"):
            visual_prompt_to_sample(prompt, label_to_category_id)

    def test_deduplicate_annotations_removes_exact_duplicates(self) -> None:
        label_id = uuid.uuid4()

        config1 = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.1), Point(x=0.5, y=0.5), Point(x=0.1, y=0.5)],
        )
        config2 = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.1), Point(x=0.5, y=0.5), Point(x=0.1, y=0.5)],
        )

        annotations = [
            AnnotationSchema(config=config1, label_id=label_id),
            AnnotationSchema(config=config2, label_id=label_id),
        ]

        result = deduplicate_annotations(annotations, 480, 640)

        assert len(result) == 1
        assert result[0].label_id == label_id

    def test_deduplicate_annotations_removes_similar_polygons(self) -> None:
        label_id = uuid.uuid4()

        # Very similar polygons (should have high IoU)
        config1 = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.1), Point(x=0.5, y=0.5), Point(x=0.1, y=0.5)],
        )
        config2 = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=0.11, y=0.11), Point(x=0.51, y=0.11), Point(x=0.51, y=0.51), Point(x=0.11, y=0.51)],
        )

        annotations = [
            AnnotationSchema(config=config1, label_id=label_id),
            AnnotationSchema(config=config2, label_id=label_id),
        ]

        result = deduplicate_annotations(annotations, 480, 640, iou_threshold=0.9)

        assert len(result) == 1

    def test_deduplicate_annotations_keeps_different_polygons(self) -> None:
        label_id = uuid.uuid4()

        # Different polygons (low IoU)
        config1 = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=0.1, y=0.1), Point(x=0.3, y=0.1), Point(x=0.3, y=0.3), Point(x=0.1, y=0.3)],
        )
        config2 = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=0.6, y=0.6), Point(x=0.8, y=0.6), Point(x=0.8, y=0.8), Point(x=0.6, y=0.8)],
        )

        annotations = [
            AnnotationSchema(config=config1, label_id=label_id),
            AnnotationSchema(config=config2, label_id=label_id),
        ]

        result = deduplicate_annotations(annotations, 480, 640)

        assert len(result) == 2

    def test_deduplicate_annotations_keeps_non_polygons(self) -> None:
        label_id = uuid.uuid4()

        polygon_config = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.1), Point(x=0.5, y=0.5), Point(x=0.1, y=0.5)],
        )
        rectangle_config = RectangleAnnotation(
            type=AnnotationType.RECTANGLE,
            points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.5)],
        )

        annotations = [
            AnnotationSchema(config=polygon_config, label_id=label_id),
            AnnotationSchema(config=rectangle_config, label_id=label_id),
        ]

        result = deduplicate_annotations(annotations, 480, 640)

        assert len(result) == 2
        assert any(ann.config.type == AnnotationType.POLYGON for ann in result)
        assert any(ann.config.type == AnnotationType.RECTANGLE for ann in result)

    def test_deduplicate_annotations_empty_list(self) -> None:
        result = deduplicate_annotations([], 480, 640)
        assert result == []

    def test_deduplicate_annotations_single_annotation(self) -> None:
        label_id = uuid.uuid4()
        config = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.1), Point(x=0.5, y=0.5), Point(x=0.1, y=0.5)],
        )
        annotations = [AnnotationSchema(config=config, label_id=label_id)]

        result = deduplicate_annotations(annotations, 480, 640)

        assert len(result) == 1
        assert result[0].label_id == label_id
