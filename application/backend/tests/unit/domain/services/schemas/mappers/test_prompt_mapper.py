# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import uuid
from types import SimpleNamespace

import numpy as np
import pytest
from getiprompt.data.base.sample import Sample

from domain.db.models import PromptType
from domain.errors import ServiceError
from domain.services.schemas.annotation import AnnotationType, Point, PolygonAnnotation, RectangleAnnotation
from domain.services.schemas.mappers.prompt import visual_prompt_to_sample


class TestPromptMapper:
    @pytest.fixture
    def sample_frame(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_visual_prompt_to_sample_with_frame(self, sample_frame):
        prompt_id = uuid.uuid4()
        project_id = uuid.uuid4()
        frame_id = uuid.uuid4()
        label_id = uuid.uuid4()

        config = PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.1), Point(x=0.5, y=0.5), Point(x=0.1, y=0.5)],
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

        result = visual_prompt_to_sample(prompt_db, frame=sample_frame)

        assert result is not None
        assert isinstance(result, Sample)
        assert np.array_equal(result.image.permute(1, 2, 0).numpy(), sample_frame)
        assert len(result.categories) == 1
        assert result.categories[0] == str(label_id)

    def test_visual_prompt_to_sample_raises_error_without_polygons(self, sample_frame):
        prompt_id = uuid.uuid4()
        frame_id = uuid.uuid4()
        label_id = uuid.uuid4()

        config = RectangleAnnotation(type=AnnotationType.RECTANGLE, points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.5)])

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

        with pytest.raises(ServiceError, match="must have at least one polygon annotation"):
            visual_prompt_to_sample(prompt_db, frame=sample_frame)

    def test_visual_prompt_to_sample_with_multiple_polygons(self, sample_frame):
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

        result = visual_prompt_to_sample(prompt_db, frame=sample_frame)

        assert result is not None
        assert isinstance(result, Sample)
        assert len(result.categories) == 2
        assert str(label_id_1) in result.categories
        assert str(label_id_2) in result.categories

    def test_visual_prompt_to_sample_raises_error_for_text_prompt(self, sample_frame):
        prompt_id = uuid.uuid4()
        project_id = uuid.uuid4()

        prompt_db = SimpleNamespace(
            id=prompt_id,
            type=PromptType.TEXT,
            text="red car",
            frame_id=None,
            project_id=project_id,
            annotations=[],
        )

        with pytest.raises(ServiceError, match="Cannot convert non-visual prompt"):
            visual_prompt_to_sample(prompt_db, frame=sample_frame)

    def test_visual_prompt_to_sample_raises_error_without_annotations(self, sample_frame):
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

        with pytest.raises(ServiceError, match="has no valid annotations"):
            visual_prompt_to_sample(prompt_db, frame=sample_frame)
