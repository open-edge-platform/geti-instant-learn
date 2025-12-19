# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from sqlalchemy.exc import IntegrityError

from domain.db.models import PromptType
from domain.errors import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
    ResourceUpdateConflictError,
    ServiceError,
)
from domain.services.prompt import PromptService
from domain.services.schemas.annotation import (
    AnnotationSchema,
    AnnotationType,
    Point,
    PolygonAnnotation,
    RectangleAnnotation,
)
from domain.services.schemas.prompt import (
    TextPromptCreateSchema,
    TextPromptUpdateSchema,
    VisualPromptCreateSchema,
    VisualPromptUpdateSchema,
)


def make_project(project_id=None, name="proj"):
    return SimpleNamespace(id=project_id or uuid.uuid4(), name=name)


def make_text_prompt_db(prompt_id=None, project_id=None, text="test prompt"):
    return SimpleNamespace(
        id=prompt_id or uuid.uuid4(),
        type=PromptType.TEXT,
        text=text,
        frame_id=None,
        project_id=project_id or uuid.uuid4(),
        annotations=[],
    )


def make_visual_prompt_db(prompt_id=None, project_id=None, frame_id=None, annotations=None):
    return SimpleNamespace(
        id=prompt_id or uuid.uuid4(),
        type=PromptType.VISUAL,
        text=None,
        frame_id=frame_id or uuid.uuid4(),
        project_id=project_id or uuid.uuid4(),
        annotations=annotations or [],
        thumbnail=None,
    )


def make_annotation_db(annotation_id=None, label_id=None):
    return SimpleNamespace(
        id=annotation_id or uuid.uuid4(),
        config={"type": "rectangle", "points": [{"x": 0.1, "y": 0.1}, {"x": 0.5, "y": 0.5}]},
        label_id=label_id,
    )


def make_label(label_id=None, project_id=None, name="car"):
    return SimpleNamespace(
        id=label_id or uuid.uuid4(),
        name=name,
        color="#FF0000",
        project_id=project_id or uuid.uuid4(),
    )


@pytest.fixture
def service():
    session = MagicMock(name="SessionMock")
    prompt_repo = MagicMock(name="PromptRepositoryMock")
    project_repo = MagicMock(name="ProjectRepositoryMock")
    frame_repo = MagicMock(name="FrameRepositoryMock")
    label_repo = MagicMock(name="LabelRepositoryMock")
    return PromptService(
        session=session,
        prompt_repository=prompt_repo,
        project_repository=project_repo,
        frame_repository=frame_repo,
        label_repository=label_repo,
    )


@pytest.fixture
def project_id():
    return uuid.uuid4()


@pytest.fixture
def frame_id():
    return uuid.uuid4()


@pytest.fixture
def label_id():
    return uuid.uuid4()


@pytest.fixture
def test_image():
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def mock_project(service, project_id):
    project = make_project(project_id)
    service.project_repository.get_by_id.return_value = project
    return project


@pytest.fixture
def mock_frame(service, project_id, frame_id, test_image):
    service.frame_repository.get_frame_path.return_value = "/path/to/frame.jpg"
    service.frame_repository.read_frame.return_value = test_image
    return test_image


@pytest.fixture
def mock_label(service, project_id, label_id):
    label = make_label(label_id=label_id, project_id=project_id)
    service.label_repository.get_by_id_and_project.return_value = label
    return label


@pytest.fixture
def setup_visual_prompt_test(mock_project, mock_frame, mock_label, project_id, frame_id, label_id):
    return project_id, frame_id, label_id


@pytest.fixture
def rectangle_annotation(label_id):
    return AnnotationSchema(
        config=RectangleAnnotation(type="rectangle", points=[Point(x=10, y=10), Point(x=50, y=50)]),
        label_id=label_id,
    )


@pytest.fixture
def polygon_annotation(label_id):
    return AnnotationSchema(
        config=PolygonAnnotation(
            type=AnnotationType.POLYGON,
            points=[Point(x=10, y=10), Point(x=50, y=10), Point(x=50, y=50), Point(x=10, y=50)],
        ),
        label_id=label_id,
    )


def test_list_prompts_success(service, mock_project, project_id, test_image):
    text_prompt = make_text_prompt_db(project_id=project_id)
    visual_prompt = make_visual_prompt_db(project_id=project_id)
    service.prompt_repository.list_with_pagination_by_project.return_value = ([text_prompt, visual_prompt], 2)
    service.frame_repository.read_frame.return_value = test_image

    result = service.list_prompts(project_id, offset=0, limit=10)

    assert len(result.prompts) == 2
    assert result.pagination.total == 2
    assert result.pagination.count == 2
    assert result.pagination.offset == 0
    assert result.pagination.limit == 10
    service.prompt_repository.list_with_pagination_by_project.assert_called_once_with(
        project_id=project_id, offset=0, limit=10
    )


def test_list_prompts_empty(service, mock_project, project_id):
    service.prompt_repository.list_with_pagination_by_project.return_value = ([], 0)

    result = service.list_prompts(project_id)

    assert result.prompts == []
    assert result.pagination.total == 0
    assert result.pagination.count == 0


def test_get_prompt_success(service, mock_project, project_id):
    prompt_id = uuid.uuid4()
    prompt = make_text_prompt_db(prompt_id=prompt_id, project_id=project_id)
    service.prompt_repository.get_by_id_and_project.return_value = prompt

    schema = service.get_prompt(project_id=project_id, prompt_id=prompt_id)

    assert schema.id == prompt_id
    assert schema.type == PromptType.TEXT
    service.prompt_repository.get_by_id_and_project.assert_called_once_with(prompt_id, project_id)


def test_get_prompt_not_found(service, mock_project, project_id):
    service.prompt_repository.get_by_id_and_project.return_value = None

    with pytest.raises(ResourceNotFoundError) as exc_info:
        service.get_prompt(project_id=project_id, prompt_id=uuid.uuid4())

    assert exc_info.value.resource_type == ResourceType.PROMPT


def test_create_text_prompt_success(service, mock_project, project_id):
    new_id = uuid.uuid4()
    service.prompt_repository.get_text_prompt_by_project.return_value = None

    create_schema = TextPromptCreateSchema(id=new_id, type=PromptType.TEXT, content="find red car")

    result = service.create_prompt(project_id=project_id, create_data=create_schema)

    assert result.id == new_id
    assert result.type == PromptType.TEXT
    service.prompt_repository.add.assert_called_once()
    service.session.commit.assert_called_once()


def test_create_text_prompt_already_exists(service, mock_project, project_id):
    existing_id = uuid.uuid4()
    existing_prompt = make_text_prompt_db(prompt_id=existing_id, project_id=project_id)
    service.prompt_repository.get_text_prompt_by_project.return_value = existing_prompt

    create_schema = TextPromptCreateSchema(id=uuid.uuid4(), type=PromptType.TEXT, content="another prompt")

    with pytest.raises(ResourceAlreadyExistsError) as exc_info:
        service.create_prompt(project_id=project_id, create_data=create_schema)

    assert exc_info.value.resource_type == ResourceType.PROMPT
    assert exc_info.value.field == "type"
    assert str(existing_id) in str(exc_info.value)
    service.prompt_repository.add.assert_not_called()


def test_create_visual_prompt_success(service, setup_visual_prompt_test, rectangle_annotation):
    project_id, frame_id, label_id = setup_visual_prompt_test
    new_id = uuid.uuid4()

    create_schema = VisualPromptCreateSchema(
        id=new_id,
        type=PromptType.VISUAL,
        frame_id=frame_id,
        annotations=[rectangle_annotation],
    )

    result = service.create_prompt(project_id=project_id, create_data=create_schema)

    assert result.id == new_id
    assert result.type == PromptType.VISUAL
    service.frame_repository.get_frame_path.assert_called_with(project_id, frame_id)
    service.frame_repository.read_frame.assert_called_once_with(project_id, frame_id)
    service.label_repository.get_by_id_and_project.assert_called_with(label_id, project_id)
    service.prompt_repository.add.assert_called_once()
    service.session.commit.assert_called_once()


def test_create_visual_prompt_deduplicates_annotations(service, setup_visual_prompt_test, polygon_annotation, caplog):
    project_id, frame_id, _ = setup_visual_prompt_test
    new_id = uuid.uuid4()

    create_schema = VisualPromptCreateSchema(
        id=new_id,
        type=PromptType.VISUAL,
        frame_id=frame_id,
        annotations=[polygon_annotation, polygon_annotation],
    )

    with caplog.at_level("INFO"):
        result = service.create_prompt(project_id=project_id, create_data=create_schema)

    assert result.id == new_id
    assert any("Removed 1 duplicate annotations" in record.message for record in caplog.records)

    call_args = service.prompt_repository.add.call_args[0][0]
    assert len(call_args.annotations) == 1
    service.session.commit.assert_called_once()


def test_create_visual_prompt_frame_not_found(service, mock_project, project_id, frame_id, label_id):
    service.frame_repository.get_frame_path.return_value = None

    create_schema = VisualPromptCreateSchema(
        id=uuid.uuid4(),
        type=PromptType.VISUAL,
        frame_id=frame_id,
        annotations=[
            AnnotationSchema(
                config=RectangleAnnotation(type="rectangle", points=[Point(x=10, y=10), Point(x=50, y=50)]),
                label_id=label_id,
            )
        ],
    )

    with pytest.raises(ResourceNotFoundError) as exc_info:
        service.create_prompt(project_id=project_id, create_data=create_schema)

    assert exc_info.value.resource_type == ResourceType.FRAME
    assert str(frame_id) in str(exc_info.value)
    service.prompt_repository.add.assert_not_called()


def test_create_visual_prompt_label_not_found(service, mock_project, mock_frame, project_id, frame_id, label_id):
    service.label_repository.get_by_id_and_project.return_value = None

    create_schema = VisualPromptCreateSchema(
        id=uuid.uuid4(),
        type=PromptType.VISUAL,
        frame_id=frame_id,
        annotations=[
            AnnotationSchema(
                config=RectangleAnnotation(type="rectangle", points=[Point(x=10, y=10), Point(x=50, y=50)]),
                label_id=label_id,
            )
        ],
    )

    with pytest.raises(ResourceNotFoundError) as exc_info:
        service.create_prompt(project_id=project_id, create_data=create_schema)

    assert exc_info.value.resource_type == ResourceType.LABEL
    assert str(label_id) in str(exc_info.value)
    service.prompt_repository.add.assert_not_called()


def test_create_prompt_integrity_error_text_duplicate(service, mock_project, project_id):
    service.prompt_repository.get_text_prompt_by_project.return_value = None

    create_schema = TextPromptCreateSchema(id=uuid.uuid4(), type=PromptType.TEXT, content="test")

    mock_error = IntegrityError("statement", "params", "orig")
    mock_error.orig = Exception("UNIQUE constraint failed: uq_single_text_prompt_per_project")
    service.session.commit.side_effect = mock_error

    with pytest.raises(ResourceAlreadyExistsError) as exc_info:
        service.create_prompt(project_id=project_id, create_data=create_schema)

    assert exc_info.value.resource_type == ResourceType.PROMPT
    assert exc_info.value.field == "type"
    service.session.rollback.assert_called_once()


def test_create_prompt_integrity_error_check_constraint(service, mock_project, project_id):
    service.prompt_repository.get_text_prompt_by_project.return_value = None

    create_schema = TextPromptCreateSchema(id=uuid.uuid4(), type=PromptType.TEXT, content="test")

    mock_error = IntegrityError("statement", "params", "orig")
    mock_error.orig = Exception("CHECK constraint failed: chk_prompt_content")
    service.session.commit.side_effect = mock_error

    with pytest.raises(ServiceError) as exc_info:
        service.create_prompt(project_id=project_id, create_data=create_schema)

    assert "text prompt must have non-empty text content" in str(exc_info.value).lower()
    service.session.rollback.assert_called_once()


def test_create_prompt_integrity_error_frame_duplicate(service, setup_visual_prompt_test, rectangle_annotation):
    project_id, frame_id, _ = setup_visual_prompt_test

    create_schema = VisualPromptCreateSchema(
        id=uuid.uuid4(),
        type=PromptType.VISUAL,
        frame_id=frame_id,
        annotations=[rectangle_annotation],
    )

    mock_error = IntegrityError("statement", "params", "orig")
    mock_error.orig = Exception("UNIQUE constraint failed: uq_unique_frame_id_per_prompt")
    service.session.commit.side_effect = mock_error

    with pytest.raises(ResourceAlreadyExistsError) as exc_info:
        service.create_prompt(project_id=project_id, create_data=create_schema)

    assert exc_info.value.resource_type == ResourceType.PROMPT
    assert exc_info.value.field == "frame_id"
    assert str(frame_id) in str(exc_info.value)
    service.session.rollback.assert_called_once()


def test_delete_text_prompt_success(service, mock_project, project_id):
    prompt_id = uuid.uuid4()
    prompt = make_text_prompt_db(prompt_id=prompt_id, project_id=project_id)
    service.prompt_repository.get_by_id_and_project.return_value = prompt

    service.delete_prompt(project_id=project_id, prompt_id=prompt_id)

    service.prompt_repository.delete.assert_called_once_with(prompt_id)
    service.session.commit.assert_called_once()


def test_delete_visual_prompt_deletes_frame(service, mock_project, project_id, frame_id):
    prompt_id = uuid.uuid4()
    prompt = make_visual_prompt_db(prompt_id=prompt_id, project_id=project_id, frame_id=frame_id)
    service.prompt_repository.get_by_id_and_project.return_value = prompt
    service.frame_repository.delete_frame.return_value = True

    service.delete_prompt(project_id=project_id, prompt_id=prompt_id)

    service.frame_repository.delete_frame.assert_called_once_with(project_id, frame_id)
    service.prompt_repository.delete.assert_called_once_with(prompt_id)
    service.session.commit.assert_called_once()


def test_delete_prompt_not_found(service, mock_project, project_id):
    service.prompt_repository.get_by_id_and_project.return_value = None

    with pytest.raises(ResourceNotFoundError):
        service.delete_prompt(project_id=project_id, prompt_id=uuid.uuid4())


def test_update_text_prompt_success(service, mock_project, project_id):
    prompt_id = uuid.uuid4()
    prompt = make_text_prompt_db(prompt_id=prompt_id, project_id=project_id, text="old")
    service.prompt_repository.get_by_id_and_project.return_value = prompt
    service.prompt_repository.update.return_value = prompt

    update_schema = TextPromptUpdateSchema(type=PromptType.TEXT, content="new content")

    result = service.update_prompt(project_id=project_id, prompt_id=prompt_id, update_data=update_schema)

    assert result.id == prompt_id
    assert prompt.text == "new content"
    service.session.commit.assert_called_once()


def test_update_visual_prompt_frame_success(service, mock_project, project_id, label_id, test_image):
    prompt_id = uuid.uuid4()
    old_frame_id = uuid.uuid4()
    new_frame_id = uuid.uuid4()

    prompt = make_visual_prompt_db(
        prompt_id=prompt_id,
        project_id=project_id,
        frame_id=old_frame_id,
        annotations=[make_annotation_db(label_id=label_id)],
    )
    service.prompt_repository.get_by_id_and_project.return_value = prompt
    service.prompt_repository.update.return_value = prompt
    service.frame_repository.delete_frame.return_value = True
    service.label_repository.get_by_id_and_project.return_value = make_label(label_id=label_id, project_id=project_id)
    service.frame_repository.read_frame.return_value = test_image

    update_schema = VisualPromptUpdateSchema(type=PromptType.VISUAL, frame_id=new_frame_id, annotations=None)

    result = service.update_prompt(project_id=project_id, prompt_id=prompt_id, update_data=update_schema)

    assert result.frame_id == new_frame_id
    service.frame_repository.delete_frame.assert_called_once_with(project_id, old_frame_id)
    service.label_repository.get_by_id_and_project.assert_called_with(label_id, project_id)
    service.session.commit.assert_called_once()


def test_update_visual_prompt_annotations_success(
    service, mock_project, project_id, frame_id, label_id, test_image, rectangle_annotation
):
    prompt_id = uuid.uuid4()

    prompt = make_visual_prompt_db(prompt_id=prompt_id, project_id=project_id, frame_id=frame_id)
    service.prompt_repository.get_by_id_and_project.return_value = prompt
    service.prompt_repository.update.return_value = prompt
    service.label_repository.get_by_id_and_project.return_value = make_label(label_id=label_id, project_id=project_id)
    service.frame_repository.read_frame.return_value = test_image

    update_schema = VisualPromptUpdateSchema(type=PromptType.VISUAL, frame_id=None, annotations=[rectangle_annotation])

    service.update_prompt(project_id=project_id, prompt_id=prompt_id, update_data=update_schema)

    assert service.label_repository.get_by_id_and_project.call_count >= 1
    assert service.frame_repository.read_frame.call_count == 2
    service.session.commit.assert_called_once()


def test_update_visual_prompt_deduplicates_annotations(
    service, mock_project, project_id, frame_id, test_image, polygon_annotation, caplog
):
    prompt_id = uuid.uuid4()

    prompt = make_visual_prompt_db(prompt_id=prompt_id, project_id=project_id, frame_id=frame_id)
    service.prompt_repository.get_by_id_and_project.return_value = prompt
    service.prompt_repository.update.return_value = prompt
    service.label_repository.get_by_id_and_project.return_value = make_label(
        label_id=polygon_annotation.label_id, project_id=project_id
    )
    service.frame_repository.read_frame.return_value = test_image

    update_schema = VisualPromptUpdateSchema(
        type=PromptType.VISUAL,
        frame_id=None,
        annotations=[polygon_annotation, polygon_annotation],
    )

    with caplog.at_level("INFO"):
        service.update_prompt(project_id=project_id, prompt_id=prompt_id, update_data=update_schema)

    assert any("Removed 1 duplicate annotations" in record.message for record in caplog.records)
    service.session.commit.assert_called_once()


def test_update_prompt_type_change_conflict(service, mock_project, project_id, label_id):
    prompt_id = uuid.uuid4()
    prompt = make_text_prompt_db(prompt_id=prompt_id, project_id=project_id)
    service.prompt_repository.get_by_id_and_project.return_value = prompt

    update_schema = VisualPromptUpdateSchema(
        type=PromptType.VISUAL,
        frame_id=uuid.uuid4(),
        annotations=[
            AnnotationSchema(
                config=RectangleAnnotation(type="rectangle", points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.5)]),
                label_id=label_id,
            )
        ],
    )

    with pytest.raises(ResourceUpdateConflictError) as exc_info:
        service.update_prompt(project_id=project_id, prompt_id=prompt_id, update_data=update_schema)

    assert exc_info.value.field == "type"
    service.session.commit.assert_not_called()


def test_update_prompt_not_found(service, mock_project, project_id):
    service.prompt_repository.get_by_id_and_project.return_value = None

    update_schema = TextPromptUpdateSchema(type=PromptType.TEXT, content="new")

    with pytest.raises(ResourceNotFoundError):
        service.update_prompt(project_id=project_id, prompt_id=uuid.uuid4(), update_data=update_schema)


def test_update_visual_prompt_new_frame_not_found(service, mock_project, project_id):
    prompt_id = uuid.uuid4()
    new_frame_id = uuid.uuid4()

    prompt = make_visual_prompt_db(prompt_id=prompt_id, project_id=project_id)
    service.prompt_repository.get_by_id_and_project.return_value = prompt
    service.frame_repository.read_frame.return_value = None

    update_schema = VisualPromptUpdateSchema(type=PromptType.VISUAL, frame_id=new_frame_id, annotations=None)

    with pytest.raises(ResourceNotFoundError) as exc_info:
        service.update_prompt(project_id=project_id, prompt_id=prompt_id, update_data=update_schema)

    assert exc_info.value.resource_type == ResourceType.FRAME
    service.session.commit.assert_not_called()


def test_project_not_found(service):
    service.project_repository.get_by_id.return_value = None

    with pytest.raises(ResourceNotFoundError) as exc_info:
        service.list_prompts(uuid.uuid4())

    assert exc_info.value.resource_type == ResourceType.PROJECT


def test_get_reference_batch_text_prompts(service, mock_project, project_id):
    batch = service.get_reference_batch(project_id, PromptType.TEXT)
    assert batch is None


def test_get_reference_batch_for_visual_prompts(service, mock_project, project_id, frame_id, label_id):
    annotation_db = SimpleNamespace(
        id=uuid.uuid4(),
        config=PolygonAnnotation(
            type="polygon",
            points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.1), Point(x=0.5, y=0.5), Point(x=0.1, y=0.5)],
        ),
        label_id=label_id,
    )
    visual_prompt = make_visual_prompt_db(project_id=project_id, frame_id=frame_id, annotations=[annotation_db])
    service.prompt_repository.list_all_by_project.return_value = [visual_prompt]

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    service.frame_repository.read_frame.return_value = frame

    result = service.get_reference_batch(project_id, PromptType.VISUAL)

    assert result is not None
    assert len(result) == 1
    service.prompt_repository.list_all_by_project.assert_called_once_with(
        project_id=project_id, prompt_type=PromptType.VISUAL
    )
    sample = result[0]

    assert sample.image is not None
    assert sample.masks is not None
    assert len(sample.masks) == 1
    assert str(label_id) in sample.categories

    service.frame_repository.read_frame.assert_called_once_with(project_id, frame_id)


def test_get_reference_batch_visual_prompts_empty(service, mock_project, project_id):
    service.prompt_repository.list_all_by_project.return_value = []

    result = service.get_reference_batch(project_id, PromptType.VISUAL)

    assert result is None


def test_get_reference_batch_visual_prompt_frame_not_found(service, mock_project, project_id, frame_id):
    visual_prompt = make_visual_prompt_db(project_id=project_id, frame_id=frame_id)
    service.prompt_repository.list_all_by_project.return_value = [visual_prompt]
    service.frame_repository.read_frame.return_value = None

    result = service.get_reference_batch(project_id, PromptType.VISUAL)

    assert result is None


def test_get_reference_batch_project_not_found(service):
    project_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = None

    batch = service.get_reference_batch(project_id, PromptType.VISUAL)
    assert batch is None


def test_get_reference_batch_visual_prompt_mapper_error_handled(service, mock_project, project_id, frame_id):
    visual_prompt = make_visual_prompt_db(project_id=project_id, frame_id=frame_id)
    service.prompt_repository.list_all_by_project.return_value = [visual_prompt]

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    service.frame_repository.read_frame.return_value = frame

    from unittest.mock import patch

    with patch("domain.services.prompt.visual_prompt_to_sample", side_effect=ServiceError("Mapper error")):
        result = service.get_reference_batch(project_id, PromptType.VISUAL)

    assert result is None


def test_normalization_scales_visual_points(service, frame_id, label_id):
    create_schema = VisualPromptCreateSchema(
        id=uuid.uuid4(),
        type=PromptType.VISUAL,
        frame_id=frame_id,
        annotations=[
            AnnotationSchema(
                config=RectangleAnnotation(
                    type=AnnotationType.RECTANGLE,
                    points=[Point(x=10.0, y=20.0), Point(x=60.0, y=120.0)],
                ),
                label_id=label_id,
            )
        ],
    )

    normalized = service._normalization(create_schema, height=200, width=100)

    assert normalized.annotations[0].config.points[0].x == pytest.approx(0.1)
    assert normalized.annotations[0].config.points[0].y == pytest.approx(0.1)
    assert normalized.annotations[0].config.points[1].x == pytest.approx(0.6)
    assert normalized.annotations[0].config.points[1].y == pytest.approx(0.6)


def test_denormalization_scales_visual_points(service, project_id, frame_id):
    annotation = make_annotation_db()
    annotation.config = {
        "type": AnnotationType.RECTANGLE,
        "points": [{"x": 0.25, "y": 0.5}, {"x": 0.75, "y": 0.9}],
    }
    prompt = make_visual_prompt_db(project_id=project_id, frame_id=frame_id, annotations=[annotation])
    service.frame_repository.read_frame.return_value = np.zeros((100, 200, 3), dtype=np.uint8)

    denormalized = service._denormalization(project_id=project_id, data=prompt)

    points = denormalized.annotations[0].config["points"]
    assert points[0]["x"] == 50
    assert points[0]["y"] == 50
    assert points[1]["x"] == 150
    assert points[1]["y"] == 90
    service.frame_repository.read_frame.assert_called_once_with(project_id=project_id, frame_id=frame_id)


def test_denormalization_raises_when_frame_missing(service, project_id, frame_id):
    prompt = make_visual_prompt_db(project_id=project_id, frame_id=frame_id, annotations=[make_annotation_db()])
    service.frame_repository.read_frame.return_value = None

    with pytest.raises(ResourceNotFoundError):
        service._denormalization(project_id=project_id, data=prompt)


def test_create_visual_prompt_with_multiple_similar_annotations_deduplicates(
    service, setup_visual_prompt_test, label_id, caplog
):
    project_id, frame_id, _ = setup_visual_prompt_test
    new_id = uuid.uuid4()

    polygon1 = PolygonAnnotation(
        type=AnnotationType.POLYGON,
        points=[Point(x=10, y=10), Point(x=50, y=10), Point(x=50, y=50), Point(x=10, y=50)],
    )
    polygon2 = PolygonAnnotation(
        type=AnnotationType.POLYGON,
        points=[Point(x=11, y=11), Point(x=51, y=11), Point(x=51, y=51), Point(x=11, y=51)],
    )
    polygon3 = PolygonAnnotation(
        type=AnnotationType.POLYGON,
        points=[Point(x=60, y=60), Point(x=90, y=60), Point(x=90, y=90), Point(x=60, y=90)],
    )

    create_schema = VisualPromptCreateSchema(
        id=new_id,
        type=PromptType.VISUAL,
        frame_id=frame_id,
        annotations=[
            AnnotationSchema(config=polygon1, label_id=label_id),
            AnnotationSchema(config=polygon2, label_id=label_id),
            AnnotationSchema(config=polygon3, label_id=label_id),
        ],
    )

    with caplog.at_level("INFO"):
        service.create_prompt(project_id=project_id, create_data=create_schema)

    assert any("Removed 1 duplicate annotations" in record.message for record in caplog.records)

    call_args = service.prompt_repository.add.call_args[0][0]
    assert len(call_args.annotations) == 2

    service.prompt_repository.add.assert_called_once()
    service.session.commit.assert_called_once()


def test_update_visual_prompt_with_similar_annotations_deduplicates(
    service, mock_project, project_id, frame_id, label_id, test_image, caplog
):
    prompt_id = uuid.uuid4()

    prompt = make_visual_prompt_db(prompt_id=prompt_id, project_id=project_id, frame_id=frame_id)
    service.prompt_repository.get_by_id_and_project.return_value = prompt
    service.prompt_repository.update.return_value = prompt
    service.label_repository.get_by_id_and_project.return_value = make_label(label_id=label_id, project_id=project_id)
    service.frame_repository.read_frame.return_value = test_image

    polygon1 = PolygonAnnotation(
        type=AnnotationType.POLYGON,
        points=[Point(x=10, y=10), Point(x=50, y=10), Point(x=50, y=50), Point(x=10, y=50)],
    )
    polygon2 = PolygonAnnotation(
        type=AnnotationType.POLYGON,
        points=[Point(x=11, y=11), Point(x=51, y=11), Point(x=51, y=51), Point(x=11, y=51)],
    )

    update_schema = VisualPromptUpdateSchema(
        type=PromptType.VISUAL,
        frame_id=None,
        annotations=[
            AnnotationSchema(config=polygon1, label_id=label_id),
            AnnotationSchema(config=polygon2, label_id=label_id),
        ],
    )

    with caplog.at_level("INFO"):
        service.update_prompt(project_id=project_id, prompt_id=prompt_id, update_data=update_schema)

    assert any("Removed 1 duplicate annotations" in record.message for record in caplog.records)
    service.session.commit.assert_called_once()


def test_create_visual_prompt_deduplication_preserves_order(
    service, mock_project, project_id, frame_id, test_image, caplog
):
    label_id_1 = uuid.uuid4()
    label_id_2 = uuid.uuid4()
    new_id = uuid.uuid4()

    service.frame_repository.get_frame_path.return_value = "/path/to/frame.jpg"
    service.frame_repository.read_frame.return_value = test_image

    label1 = make_label(label_id=label_id_1, project_id=project_id, name="car")
    label2 = make_label(label_id=label_id_2, project_id=project_id, name="person")
    service.label_repository.get_by_id_and_project.side_effect = (
        lambda lid, pid: label1 if lid == label_id_1 else label2
    )

    polygon1 = PolygonAnnotation(
        type=AnnotationType.POLYGON,
        points=[Point(x=10, y=10), Point(x=30, y=10), Point(x=30, y=30), Point(x=10, y=30)],
    )
    polygon2 = PolygonAnnotation(
        type=AnnotationType.POLYGON,
        points=[Point(x=10, y=10), Point(x=30, y=10), Point(x=30, y=30), Point(x=10, y=30)],
    )
    polygon3 = PolygonAnnotation(
        type=AnnotationType.POLYGON,
        points=[Point(x=60, y=60), Point(x=80, y=60), Point(x=80, y=80), Point(x=60, y=80)],
    )

    create_schema = VisualPromptCreateSchema(
        id=new_id,
        type=PromptType.VISUAL,
        frame_id=frame_id,
        annotations=[
            AnnotationSchema(config=polygon1, label_id=label_id_1),
            AnnotationSchema(config=polygon2, label_id=label_id_1),
            AnnotationSchema(config=polygon3, label_id=label_id_2),
        ],
    )

    with caplog.at_level("INFO"):
        service.create_prompt(project_id=project_id, create_data=create_schema)

    assert any("Removed 1 duplicate annotations" in record.message for record in caplog.records)

    call_args = service.prompt_repository.add.call_args[0][0]
    assert len(call_args.annotations) == 2

    service.prompt_repository.add.assert_called_once()
    service.session.commit.assert_called_once()


def test_create_visual_prompt_deduplication_only_affects_polygons(service, setup_visual_prompt_test, label_id, caplog):
    project_id, frame_id, _ = setup_visual_prompt_test
    new_id = uuid.uuid4()

    polygon = PolygonAnnotation(
        type=AnnotationType.POLYGON,
        points=[Point(x=10, y=10), Point(x=30, y=10), Point(x=30, y=30), Point(x=10, y=30)],
    )
    rectangle = RectangleAnnotation(type=AnnotationType.RECTANGLE, points=[Point(x=50, y=50), Point(x=80, y=80)])

    create_schema = VisualPromptCreateSchema(
        id=new_id,
        type=PromptType.VISUAL,
        frame_id=frame_id,
        annotations=[
            AnnotationSchema(config=polygon, label_id=label_id),
            AnnotationSchema(config=rectangle, label_id=label_id),
        ],
    )

    with caplog.at_level("INFO"):
        service.create_prompt(project_id=project_id, create_data=create_schema)

    assert not any("Removed" in record.message and "duplicate" in record.message for record in caplog.records)

    call_args = service.prompt_repository.add.call_args[0][0]
    assert len(call_args.annotations) == 2

    service.prompt_repository.add.assert_called_once()
    service.session.commit.assert_called_once()


def test_update_visual_prompt_single_frame_read_for_normalization_and_deduplication(
    service, mock_project, project_id, frame_id, label_id, test_image
):
    prompt_id = uuid.uuid4()

    prompt = make_visual_prompt_db(prompt_id=prompt_id, project_id=project_id, frame_id=frame_id)
    service.prompt_repository.get_by_id_and_project.return_value = prompt
    service.prompt_repository.update.return_value = prompt
    service.label_repository.get_by_id_and_project.return_value = make_label(label_id=label_id, project_id=project_id)
    service.frame_repository.read_frame.return_value = test_image

    polygon = PolygonAnnotation(
        type=AnnotationType.POLYGON,
        points=[Point(x=10, y=10), Point(x=50, y=10), Point(x=50, y=50), Point(x=10, y=50)],
    )

    update_schema = VisualPromptUpdateSchema(
        type=PromptType.VISUAL,
        frame_id=None,
        annotations=[AnnotationSchema(config=polygon, label_id=label_id)],
    )

    service.update_prompt(project_id=project_id, prompt_id=prompt_id, update_data=update_schema)

    assert service.frame_repository.read_frame.call_count == 2
    service.session.commit.assert_called_once()
