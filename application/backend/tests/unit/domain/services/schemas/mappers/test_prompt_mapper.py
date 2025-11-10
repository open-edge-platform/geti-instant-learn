# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from domain.db.models import PromptType
from domain.services.schemas.annotation import Point, PolygonAnnotation, RectangleAnnotation
from domain.services.schemas.mappers.prompt import prompt_db_to_training_sample
from domain.services.schemas.prompt import TextTrainingSample, VisualTrainingSample


class TestPromptMapper:
    @pytest.fixture
    def sample_frame(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_text_prompt_to_training_sample_with_content(self):
        prompt_id = uuid.uuid4()
        project_id = uuid.uuid4()
        text_content = "red car"

        prompt_db = SimpleNamespace(
            id=prompt_id,
            type=PromptType.TEXT,
            text=text_content,
            frame_id=None,
            project_id=project_id,
            annotations=[],
        )

        result = prompt_db_to_training_sample(prompt_db)

        assert result is not None
        assert isinstance(result, TextTrainingSample)
        assert result.content == text_content

    def test_visual_prompt_to_training_sample_with_frame(self, sample_frame):
        prompt_id = uuid.uuid4()
        project_id = uuid.uuid4()
        frame_id = uuid.uuid4()
        label_id = uuid.uuid4()

        config = RectangleAnnotation(type="rectangle", points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.5)])

        annotation_db = SimpleNamespace(
            id=uuid.uuid4(),
            config=config,
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

        result = prompt_db_to_training_sample(prompt_db, frame=sample_frame)

        assert result is not None
        assert isinstance(result, VisualTrainingSample)
        assert np.array_equal(result.frame, sample_frame)
        assert len(result.annotations) == 1
        assert result.annotations[0].label_id == label_id
        assert result.annotations[0].config == config

    def test_visual_prompt_to_training_sample_without_frame(self):
        prompt_id = uuid.uuid4()
        frame_id = uuid.uuid4()
        label_id = uuid.uuid4()

        config = RectangleAnnotation(type="rectangle", points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.5)])

        annotation_db = SimpleNamespace(
            id=uuid.uuid4(),
            config=config,
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

        result = prompt_db_to_training_sample(prompt_db, frame=None)

        assert result is None

    def test_visual_prompt_with_multiple_annotations(self, sample_frame):
        prompt_id = uuid.uuid4()
        frame_id = uuid.uuid4()
        label_id_1 = uuid.uuid4()
        label_id_2 = uuid.uuid4()

        config_1 = RectangleAnnotation(type="rectangle", points=[Point(x=0.1, y=0.1), Point(x=0.3, y=0.3)])
        config_2 = PolygonAnnotation(
            type="polygon", points=[Point(x=0.5, y=0.5), Point(x=0.7, y=0.7), Point(x=0.6, y=0.8)]
        )

        annotation_db_1 = SimpleNamespace(
            id=uuid.uuid4(),
            config=config_1,
            label_id=label_id_1,
            prompt_id=prompt_id,
        )
        annotation_db_2 = SimpleNamespace(
            id=uuid.uuid4(),
            config=config_2,
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

        result = prompt_db_to_training_sample(prompt_db, frame=sample_frame)

        assert result is not None
        assert isinstance(result, VisualTrainingSample)
        assert len(result.annotations) == 2
        assert result.annotations[0].label_id == label_id_1
        assert result.annotations[1].label_id == label_id_2
        assert result.annotations[0].config == config_1
        assert result.annotations[1].config == config_2
