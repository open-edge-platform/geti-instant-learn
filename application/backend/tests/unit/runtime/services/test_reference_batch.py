# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from instantlearn.data.base.sample import Sample

from domain.db.models import PromptType
from domain.errors import ServiceError
from domain.services.schemas.annotation import AnnotationType, Point, PolygonAnnotation, RectangleAnnotation
from domain.services.schemas.pipeline import PipelineConfig
from domain.services.schemas.processor import MatcherConfig, Sam3Config
from runtime.services.reference_batch import ReferenceBatchService, visual_prompt_to_sample


class FakeSessionCtx:
    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def session_factory():
    factory = Mock()
    factory.return_value = FakeSessionCtx(Mock())
    return factory


@pytest.fixture
def frame_repository():
    return Mock()


@pytest.fixture
def service(session_factory, frame_repository):
    return ReferenceBatchService(session_factory, frame_repository)


@pytest.fixture
def sample_frame() -> np.ndarray:
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


class TestVisualPromptToSample:
    def test_with_frame(self, sample_frame: np.ndarray) -> None:
        prompt_id = uuid.uuid4()
        project_id = uuid.uuid4()
        frame_id = uuid.uuid4()
        label_id = uuid.uuid4()

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

    def test_raises_error_without_polygons(self, sample_frame: np.ndarray) -> None:
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

    def test_with_multiple_polygons(self, sample_frame: np.ndarray) -> None:
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

    def test_raises_error_for_text_prompt(self, sample_frame: np.ndarray) -> None:
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

    def test_raises_error_without_annotations(self, sample_frame: np.ndarray) -> None:
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


class TestReferenceBatchServiceBuild:
    def test_build_returns_none_when_no_processor(self, service):
        cfg = PipelineConfig(project_id=uuid.uuid4(), processor=None)
        result = service.build(cfg)
        assert result is None

    def test_build_returns_none_for_sam3_text_mode(self, service):
        cfg = PipelineConfig(project_id=uuid.uuid4(), processor=Sam3Config(), prompt_mode=PromptType.TEXT)
        result = service.build(cfg)
        assert result is None

    def test_build_returns_none_when_no_prompts(self, service):
        cfg = PipelineConfig(project_id=uuid.uuid4(), processor=MatcherConfig(), prompt_mode=PromptType.VISUAL)
        with (
            patch("runtime.services.reference_batch.PromptRepository") as prompt_repo_cls,
            patch("runtime.services.reference_batch.LabelService"),
        ):
            prompt_repo_cls.return_value.list_by_project_and_type.return_value = []

            result = service.build(cfg)

        assert result is None

    def test_build_returns_batch_for_visual_prompts(self, service, frame_repository):
        project_id = uuid.uuid4()
        frame_id = uuid.uuid4()
        label_id = uuid.uuid4()

        cfg = PipelineConfig(project_id=uuid.uuid4(), processor=MatcherConfig(), prompt_mode=PromptType.VISUAL)

        annotation_db = SimpleNamespace(
            id=uuid.uuid4(),
            config={
                "type": "polygon",
                "points": [{"x": 10, "y": 10}, {"x": 50, "y": 10}, {"x": 50, "y": 50}, {"x": 10, "y": 50}],
            },
            label_id=label_id,
        )
        prompt_db = SimpleNamespace(
            id=uuid.uuid4(),
            type=PromptType.VISUAL,
            text=None,
            frame_id=frame_id,
            project_id=project_id,
            annotations=[annotation_db],
        )

        frame_bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        frame_repository.read_frame.return_value = frame_bgr

        fake_sample = MagicMock(name="Sample")
        fake_batch = MagicMock(name="Batch")
        fake_batch.samples = [fake_sample]

        with (
            patch("runtime.services.reference_batch.PromptRepository") as prompt_repo_cls,
            patch("runtime.services.reference_batch.LabelService") as label_svc_cls,
            patch("runtime.services.reference_batch.cv2.cvtColor", return_value=np.zeros((64, 64, 3), dtype=np.uint8)),
            patch("runtime.services.reference_batch.visual_prompt_to_sample", return_value=fake_sample),
            patch("runtime.services.reference_batch.Batch.collate", return_value=fake_batch),
        ):
            prompt_repo_cls.return_value.list_by_project_and_type.return_value = [prompt_db]
            label_svc_cls.return_value.build_category_mappings.return_value = SimpleNamespace(
                label_to_category_id={label_id: 0},
                category_id_to_label_id={0: str(label_id)},
            )
            label_svc_cls.return_value.get_label_names.return_value = {label_id: "car"}

            result = service.build(cfg)

        assert result is not None
        batch, category_mapping = result
        assert batch is fake_batch
        assert category_mapping == {0: str(label_id)}

    def test_build_skips_prompt_with_missing_frame(self, service, frame_repository):
        project_id = uuid.uuid4()
        label_id = uuid.uuid4()

        cfg = PipelineConfig(project_id=uuid.uuid4(), processor=MatcherConfig(), prompt_mode=PromptType.VISUAL)

        prompt_db = SimpleNamespace(
            id=uuid.uuid4(),
            type=PromptType.VISUAL,
            text=None,
            frame_id=uuid.uuid4(),
            project_id=project_id,
            annotations=[SimpleNamespace(id=uuid.uuid4(), config={"type": "polygon", "points": []}, label_id=label_id)],
        )

        frame_repository.read_frame.return_value = None

        with (
            patch("runtime.services.reference_batch.PromptRepository") as prompt_repo_cls,
            patch("runtime.services.reference_batch.LabelService") as label_svc_cls,
        ):
            prompt_repo_cls.return_value.list_by_project_and_type.return_value = [prompt_db]
            label_svc_cls.return_value.build_category_mappings.return_value = SimpleNamespace(
                label_to_category_id={}, category_id_to_label_id={}
            )
            label_svc_cls.return_value.get_label_names.return_value = {}

            result = service.build(cfg)

        assert result is None
