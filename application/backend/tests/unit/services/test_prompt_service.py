# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from db.models import PromptType
from exceptions.custom_errors import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
    ResourceUpdateConflictError,
    ServiceError,
)
from services.prompt import PromptService
from services.schemas.annotation import AnnotationSchema, Point, RectangleAnnotation
from services.schemas.prompt import (
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
    )


def make_annotation_db(annotation_id=None, label_id=None):
    return SimpleNamespace(
        id=annotation_id or uuid.uuid4(),
        config={"type": "point", "x": 0.5, "y": 0.5},
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


def test_list_prompts_success(service):
    project_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    text_prompt = make_text_prompt_db(project_id=project_id)
    visual_prompt = make_visual_prompt_db(project_id=project_id)
    service.prompt_repository.get_paginated.return_value = ([text_prompt, visual_prompt], 2)

    result = service.list_prompts(project_id, offset=0, limit=10)

    assert len(result.prompts) == 2
    assert result.pagination.total == 2
    service.prompt_repository.get_paginated.assert_called_once_with(project_id, offset=0, limit=10)


def test_list_prompts_empty(service):
    project_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    service.prompt_repository.get_paginated.return_value = ([], 0)

    result = service.list_prompts(project_id)

    assert result.prompts == []
    assert result.pagination.total == 0


def test_get_prompt_success(service):
    project_id = uuid.uuid4()
    prompt_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    prompt = make_text_prompt_db(prompt_id=prompt_id, project_id=project_id)
    service.prompt_repository.get_by_id_and_project.return_value = prompt

    schema = service.get_prompt(project_id=project_id, prompt_id=prompt_id)

    assert schema.id == prompt_id
    assert schema.type == PromptType.TEXT
    service.prompt_repository.get_by_id_and_project.assert_called_once_with(prompt_id=prompt_id, project_id=project_id)


def test_get_prompt_not_found(service):
    project_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    service.prompt_repository.get_by_id_and_project.return_value = None

    with pytest.raises(ResourceNotFoundError) as exc_info:
        service.get_prompt(project_id=project_id, prompt_id=uuid.uuid4())

    assert exc_info.value.resource_type == ResourceType.PROMPT


def test_create_text_prompt_success(service):
    new_id = uuid.uuid4()
    project_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    service.prompt_repository.get_text_prompt_by_project.return_value = None

    create_schema = TextPromptCreateSchema(
        id=new_id,
        type=PromptType.TEXT,
        content="find red car",
    )

    result = service.create_prompt(project_id=project_id, create_data=create_schema)

    assert result.id == new_id
    assert result.type == PromptType.TEXT
    service.prompt_repository.add.assert_called_once()
    service.session.commit.assert_called_once()


def test_create_text_prompt_already_exists(service):
    project_id = uuid.uuid4()
    existing_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    existing_prompt = make_text_prompt_db(prompt_id=existing_id, project_id=project_id)
    service.prompt_repository.get_text_prompt_by_project.return_value = existing_prompt

    create_schema = TextPromptCreateSchema(
        id=uuid.uuid4(),
        type=PromptType.TEXT,
        content="another prompt",
    )

    with pytest.raises(ResourceAlreadyExistsError) as exc_info:
        service.create_prompt(project_id=project_id, create_data=create_schema)

    assert exc_info.value.resource_type == ResourceType.PROMPT
    assert exc_info.value.field == "type"
    assert str(existing_id) in str(exc_info.value)
    service.prompt_repository.add.assert_not_called()


def test_create_visual_prompt_success(service):
    new_id = uuid.uuid4()
    project_id = uuid.uuid4()
    frame_id = uuid.uuid4()
    label_id = uuid.uuid4()

    service.project_repository.get_by_id.return_value = make_project(project_id)
    service.frame_repository.get_frame_path.return_value = "/path/to/frame.jpg"
    service.label_repository.get_by_id.return_value = make_label(label_id=label_id, project_id=project_id)

    create_schema = VisualPromptCreateSchema(
        id=new_id,
        type=PromptType.VISUAL,
        frame_id=frame_id,
        annotations=[
            AnnotationSchema(
                config=RectangleAnnotation(type="rectangle", points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.5)]),
                label_id=label_id,
            )
        ],
    )

    result = service.create_prompt(project_id=project_id, create_data=create_schema)

    assert result.id == new_id
    assert result.type == PromptType.VISUAL
    service.frame_repository.get_frame_path.assert_called_once_with(project_id, frame_id)
    service.label_repository.get_by_id.assert_called_once_with(project_id, label_id)
    service.prompt_repository.add.assert_called_once()
    service.session.commit.assert_called_once()


def test_create_visual_prompt_frame_not_found(service):
    project_id = uuid.uuid4()
    frame_id = uuid.uuid4()

    service.project_repository.get_by_id.return_value = make_project(project_id)
    service.frame_repository.get_frame_path.return_value = None

    create_schema = VisualPromptCreateSchema(
        id=uuid.uuid4(),
        type=PromptType.VISUAL,
        frame_id=frame_id,
        annotations=[
            AnnotationSchema(
                config=RectangleAnnotation(type="rectangle", points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.5)]),
                label_id=None,
            )
        ],
    )

    with pytest.raises(ResourceNotFoundError) as exc_info:
        service.create_prompt(project_id=project_id, create_data=create_schema)

    assert exc_info.value.resource_type == ResourceType.FRAME
    assert str(frame_id) in str(exc_info.value)
    service.prompt_repository.add.assert_not_called()


def test_create_visual_prompt_label_not_found(service):
    project_id = uuid.uuid4()
    frame_id = uuid.uuid4()
    label_id = uuid.uuid4()

    service.project_repository.get_by_id.return_value = make_project(project_id)
    service.frame_repository.get_frame_path.return_value = "/path/to/frame.jpg"
    service.label_repository.get_by_id.return_value = None

    create_schema = VisualPromptCreateSchema(
        id=uuid.uuid4(),
        type=PromptType.VISUAL,
        frame_id=frame_id,
        annotations=[
            AnnotationSchema(
                config=RectangleAnnotation(type="rectangle", points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.5)]),
                label_id=label_id,
            )
        ],
    )

    with pytest.raises(ResourceNotFoundError) as exc_info:
        service.create_prompt(project_id=project_id, create_data=create_schema)

    assert exc_info.value.resource_type == ResourceType.LABEL
    assert str(label_id) in str(exc_info.value)
    service.prompt_repository.add.assert_not_called()


def test_create_prompt_integrity_error_text_duplicate(service):
    project_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    service.prompt_repository.get_text_prompt_by_project.return_value = None

    create_schema = TextPromptCreateSchema(
        id=uuid.uuid4(),
        type=PromptType.TEXT,
        content="test",
    )

    mock_error = IntegrityError("statement", "params", "orig")
    mock_error.orig = Exception("UNIQUE constraint failed: uq_single_text_prompt_per_project")
    service.session.commit.side_effect = mock_error

    with pytest.raises(ResourceAlreadyExistsError) as exc_info:
        service.create_prompt(project_id=project_id, create_data=create_schema)

    assert exc_info.value.resource_type == ResourceType.PROMPT
    assert exc_info.value.field == "type"
    service.session.rollback.assert_called_once()


def test_create_prompt_integrity_error_check_constraint(service):
    project_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    service.prompt_repository.get_text_prompt_by_project.return_value = None

    create_schema = TextPromptCreateSchema(
        id=uuid.uuid4(),
        type=PromptType.TEXT,
        content="test",
    )

    mock_error = IntegrityError("statement", "params", "orig")
    mock_error.orig = Exception("CHECK constraint failed: chk_prompt_content")
    service.session.commit.side_effect = mock_error

    with pytest.raises(ServiceError) as exc_info:
        service.create_prompt(project_id=project_id, create_data=create_schema)

    assert "text prompt must have non-empty text content" in str(exc_info.value).lower()
    service.session.rollback.assert_called_once()


def test_delete_text_prompt_success(service):
    project_id = uuid.uuid4()
    prompt_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    prompt = make_text_prompt_db(prompt_id=prompt_id, project_id=project_id)
    service.prompt_repository.get_by_id_and_project.return_value = prompt

    service.delete_prompt(project_id=project_id, prompt_id=prompt_id)

    service.prompt_repository.delete.assert_called_once_with(prompt)
    service.session.commit.assert_called_once()


def test_delete_visual_prompt_deletes_frame(service):
    project_id = uuid.uuid4()
    prompt_id = uuid.uuid4()
    frame_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    prompt = make_visual_prompt_db(prompt_id=prompt_id, project_id=project_id, frame_id=frame_id)
    service.prompt_repository.get_by_id_and_project.return_value = prompt
    service.frame_repository.delete_frame.return_value = True

    service.delete_prompt(project_id=project_id, prompt_id=prompt_id)

    service.frame_repository.delete_frame.assert_called_once_with(project_id, frame_id)
    service.prompt_repository.delete.assert_called_once_with(prompt)
    service.session.commit.assert_called_once()


def test_delete_prompt_not_found(service):
    project_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    service.prompt_repository.get_by_id_and_project.return_value = None

    with pytest.raises(ResourceNotFoundError):
        service.delete_prompt(project_id=project_id, prompt_id=uuid.uuid4())


def test_update_text_prompt_success(service):
    project_id = uuid.uuid4()
    prompt_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    prompt = make_text_prompt_db(prompt_id=prompt_id, project_id=project_id, text="old")
    service.prompt_repository.get_by_id_and_project.return_value = prompt

    update_schema = TextPromptUpdateSchema(
        type=PromptType.TEXT,
        content="new content",
    )

    result = service.update_prompt(project_id=project_id, prompt_id=prompt_id, update_data=update_schema)

    assert result.id == prompt_id
    assert prompt.text == "new content"
    service.session.commit.assert_called_once()


def test_update_visual_prompt_frame_success(service):
    project_id = uuid.uuid4()
    prompt_id = uuid.uuid4()
    old_frame_id = uuid.uuid4()
    new_frame_id = uuid.uuid4()

    service.project_repository.get_by_id.return_value = make_project(project_id)
    prompt = make_visual_prompt_db(prompt_id=prompt_id, project_id=project_id, frame_id=old_frame_id)
    service.prompt_repository.get_by_id_and_project.return_value = prompt
    service.frame_repository.get_frame_path.return_value = "/path/to/new_frame.jpg"
    service.frame_repository.delete_frame.return_value = True

    update_schema = VisualPromptUpdateSchema(
        type=PromptType.VISUAL,
        frame_id=new_frame_id,
        annotations=None,
    )

    result = service.update_prompt(project_id=project_id, prompt_id=prompt_id, update_data=update_schema)

    assert result.frame_id == new_frame_id
    service.frame_repository.get_frame_path.assert_called_once_with(project_id, new_frame_id)
    service.frame_repository.delete_frame.assert_called_once_with(project_id, old_frame_id)
    service.session.commit.assert_called_once()


def test_update_visual_prompt_annotations_success(service):
    project_id = uuid.uuid4()
    prompt_id = uuid.uuid4()
    label_id = uuid.uuid4()

    service.project_repository.get_by_id.return_value = make_project(project_id)
    prompt = make_visual_prompt_db(prompt_id=prompt_id, project_id=project_id)
    service.prompt_repository.get_by_id_and_project.return_value = prompt
    service.label_repository.get_by_id.return_value = make_label(label_id=label_id, project_id=project_id)

    update_schema = VisualPromptUpdateSchema(
        type=PromptType.VISUAL,
        frame_id=None,
        annotations=[
            AnnotationSchema(
                config=RectangleAnnotation(type="rectangle", points=[Point(x=0.2, y=0.2), Point(x=0.7, y=0.7)]),
                label_id=label_id,
            )
        ],
    )

    service.update_prompt(project_id=project_id, prompt_id=prompt_id, update_data=update_schema)

    service.label_repository.get_by_id.assert_called_once_with(project_id, label_id)
    service.session.commit.assert_called_once()


def test_update_prompt_type_change_conflict(service):
    project_id = uuid.uuid4()
    prompt_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    prompt = make_text_prompt_db(prompt_id=prompt_id, project_id=project_id)
    service.prompt_repository.get_by_id_and_project.return_value = prompt

    update_schema = VisualPromptUpdateSchema(
        type=PromptType.VISUAL,
        frame_id=uuid.uuid4(),
        annotations=[
            AnnotationSchema(
                config=RectangleAnnotation(type="rectangle", points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.5)]),
                label_id=None,
            )
        ],
    )

    with pytest.raises(ResourceUpdateConflictError) as exc_info:
        service.update_prompt(project_id=project_id, prompt_id=prompt_id, update_data=update_schema)

    assert exc_info.value.field == "type"
    service.session.commit.assert_not_called()


def test_update_prompt_not_found(service):
    project_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    service.prompt_repository.get_by_id_and_project.return_value = None

    update_schema = TextPromptUpdateSchema(
        type=PromptType.TEXT,
        content="new",
    )

    with pytest.raises(ResourceNotFoundError):
        service.update_prompt(project_id=project_id, prompt_id=uuid.uuid4(), update_data=update_schema)


def test_update_visual_prompt_new_frame_not_found(service):
    project_id = uuid.uuid4()
    prompt_id = uuid.uuid4()
    new_frame_id = uuid.uuid4()

    service.project_repository.get_by_id.return_value = make_project(project_id)
    prompt = make_visual_prompt_db(prompt_id=prompt_id, project_id=project_id)
    service.prompt_repository.get_by_id_and_project.return_value = prompt
    service.frame_repository.get_frame_path.return_value = None

    update_schema = VisualPromptUpdateSchema(
        type=PromptType.VISUAL,
        frame_id=new_frame_id,
        annotations=None,
    )

    with pytest.raises(ResourceNotFoundError) as exc_info:
        service.update_prompt(project_id=project_id, prompt_id=prompt_id, update_data=update_schema)

    assert exc_info.value.resource_type == ResourceType.FRAME
    service.session.commit.assert_not_called()


def test_project_not_found(service):
    service.project_repository.get_by_id.return_value = None

    with pytest.raises(ResourceNotFoundError) as exc_info:
        service.list_prompts(uuid.uuid4())

    assert exc_info.value.resource_type == ResourceType.PROJECT
