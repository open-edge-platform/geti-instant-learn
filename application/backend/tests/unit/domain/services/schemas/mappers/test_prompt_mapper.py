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


class TestPromptMapper:
    @pytest.fixture
    def sample_frame(self) -> np.ndarray:
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_visual_prompt_to_sample_with_frame(self, sample_frame: np.ndarray) -> None:
        prompt_id = uuid.uuid4()
        project_id = uuid.uuid4()
        frame_id = uuid.uuid4()
        label_id = uuid.uuid4()

        # Use pixel coordinates for the 480x640 frame
        config = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=64, y=48), Point(x=320, y=48), Point(x=320, y=240), Point(x=64, y=240)],
        )

        annotation_db = SimpleNamespace(
            id=uuid.uuid4(),
            config=config.model_dump(),
            label_id=label_id,
            prompt_id=prompt_id,
        )

        prompt_db = SimpleNamespace(
            id=prompt_id,
            type=PromptType.VISUAL,
            text=None,
            frame_id=frame_id,
            project_id=project_id,
            annotations=[annotation_db],
        )

        label_to_category_id = {label_id: 0}
        label_id_to_name = {label_id: "car"}
        label_shot_counts: dict[uuid.UUID, int] = {}

        result = visual_prompt_to_sample(
            prompt_db,
            frame=sample_frame,
            label_to_category_id=label_to_category_id,
            label_id_to_name=label_id_to_name,
            label_shot_counts=label_shot_counts,
        )

        assert result is not None
        assert isinstance(result, Sample)
        assert np.array_equal(result.image.permute(1, 2, 0).numpy(), sample_frame)
        assert len(result.categories) == 1
        assert result.categories[0] == "car"
        assert label_shot_counts[label_id] == 1

    def test_visual_prompt_to_sample_raises_error_without_polygons(self, sample_frame: np.ndarray) -> None:
        prompt_id = uuid.uuid4()
        frame_id = uuid.uuid4()
        label_id = uuid.uuid4()

        config = RectangleAnnotation(type=AnnotationType.RECTANGLE, points=[Point(x=10, y=10), Point(x=50, y=50)])

        annotation_db = SimpleNamespace(
            id=uuid.uuid4(),
            config=config.model_dump(),
            label_id=label_id,
            prompt_id=prompt_id,
        )

        prompt_db = SimpleNamespace(
            id=prompt_id,
            type=PromptType.VISUAL,
            text=None,
            frame_id=frame_id,
            project_id=uuid.uuid4(),
            annotations=[annotation_db],
        )

        label_to_category_id = {label_id: 0}
        label_id_to_name = {label_id: "car"}
        label_shot_counts: dict[uuid.UUID, int] = {}

        with pytest.raises(ServiceError, match="must have at least one polygon annotation"):
            visual_prompt_to_sample(
                prompt_db,
                frame=sample_frame,
                label_to_category_id=label_to_category_id,
                label_id_to_name=label_id_to_name,
                label_shot_counts=label_shot_counts,
            )

    def test_visual_prompt_to_sample_with_multiple_polygons(self, sample_frame: np.ndarray) -> None:
        prompt_id = uuid.uuid4()
        frame_id = uuid.uuid4()
        label_id_1 = uuid.uuid4()
        label_id_2 = uuid.uuid4()

        config_1 = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=64, y=48), Point(x=192, y=48), Point(x=192, y=192), Point(x=64, y=192)],
        )
        config_2 = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=320, y=240), Point(x=448, y=336), Point(x=384, y=384)],
        )

        annotation_db_1 = SimpleNamespace(
            id=uuid.uuid4(),
            config=config_1.model_dump(),
            label_id=label_id_1,
            prompt_id=prompt_id,
        )
        annotation_db_2 = SimpleNamespace(
            id=uuid.uuid4(),
            config=config_2.model_dump(),
            label_id=label_id_2,
            prompt_id=prompt_id,
        )

        prompt_db = SimpleNamespace(
            id=prompt_id,
            type=PromptType.VISUAL,
            text=None,
            frame_id=frame_id,
            project_id=uuid.uuid4(),
            annotations=[annotation_db_1, annotation_db_2],
        )

        label_to_category_id = {label_id_1: 0, label_id_2: 1}
        label_id_to_name = {label_id_1: "car", label_id_2: "person"}
        label_shot_counts: dict[uuid.UUID, int] = {}

        result = visual_prompt_to_sample(
            prompt_db,
            frame=sample_frame,
            label_to_category_id=label_to_category_id,
            label_id_to_name=label_id_to_name,
            label_shot_counts=label_shot_counts,
        )

        assert result is not None
        assert isinstance(result, Sample)
        assert len(result.categories) == 2
        assert "car" in result.categories
        assert "person" in result.categories
        assert label_shot_counts[label_id_1] == 1
        assert label_shot_counts[label_id_2] == 1

    def test_visual_prompt_to_sample_raises_error_for_text_prompt(self, sample_frame: np.ndarray) -> None:
        prompt_id = uuid.uuid4()
        project_id = uuid.uuid4()

        prompt_db = SimpleNamespace(
            id=prompt_id,
            type=PromptType.TEXT,
            text="hello",
            frame_id=None,
            project_id=project_id,
            annotations=[],
        )

        label_to_category_id: dict[uuid.UUID, int] = {}
        label_id_to_name: dict[uuid.UUID, str] = {}
        label_shot_counts: dict[uuid.UUID, int] = {}

        with pytest.raises(ServiceError, match="Cannot convert non-visual prompt"):
            visual_prompt_to_sample(
                prompt_db,
                frame=sample_frame,
                label_to_category_id=label_to_category_id,
                label_id_to_name=label_id_to_name,
                label_shot_counts=label_shot_counts,
            )

    def test_visual_prompt_to_sample_raises_error_without_annotations(self, sample_frame: np.ndarray) -> None:
        prompt_id = uuid.uuid4()
        frame_id = uuid.uuid4()

        prompt_db = SimpleNamespace(
            id=prompt_id,
            type=PromptType.VISUAL,
            text=None,
            frame_id=frame_id,
            project_id=uuid.uuid4(),
            annotations=[],
        )

        label_to_category_id: dict[uuid.UUID, int] = {}
        label_id_to_name: dict[uuid.UUID, str] = {}
        label_shot_counts: dict[uuid.UUID, int] = {}

        with pytest.raises(ServiceError, match="has no valid annotations"):
            visual_prompt_to_sample(
                prompt_db,
                frame=sample_frame,
                label_to_category_id=label_to_category_id,
                label_id_to_name=label_id_to_name,
                label_shot_counts=label_shot_counts,
            )

    def test_deduplicate_annotations_removes_exact_duplicates(self) -> None:
        label_id = uuid.uuid4()

        config1 = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=64, y=48), Point(x=320, y=48), Point(x=320, y=240), Point(x=64, y=240)],
        )
        config2 = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=64, y=48), Point(x=320, y=48), Point(x=320, y=240), Point(x=64, y=240)],
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

        config1 = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=64, y=48), Point(x=320, y=48), Point(x=320, y=240), Point(x=64, y=240)],
        )
        config2 = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=70, y=53), Point(x=326, y=53), Point(x=326, y=246), Point(x=70, y=246)],
        )

        annotations = [
            AnnotationSchema(config=config1, label_id=label_id),
            AnnotationSchema(config=config2, label_id=label_id),
        ]

        result = deduplicate_annotations(annotations, 480, 640, iou_threshold=0.9)

        assert len(result) == 1

    def test_deduplicate_annotations_keeps_different_polygons(self) -> None:
        label_id = uuid.uuid4()

        config1 = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=64, y=48), Point(x=192, y=48), Point(x=192, y=192), Point(x=64, y=192)],
        )
        config2 = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=384, y=240), Point(x=512, y=240), Point(x=512, y=384), Point(x=384, y=384)],
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
            points=[Point(x=64, y=48), Point(x=320, y=48), Point(x=320, y=240), Point(x=64, y=240)],
        )
        rectangle_config = RectangleAnnotation(
            type=AnnotationType.RECTANGLE,
            points=[Point(x=10, y=10), Point(x=50, y=50)],
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
            points=[Point(x=64, y=48), Point(x=320, y=48), Point(x=320, y=240), Point(x=64, y=240)],
        )
        annotations = [AnnotationSchema(config=config, label_id=label_id)]

        result = deduplicate_annotations(annotations, 480, 640)

        assert len(result) == 1
        assert result[0].label_id == label_id
