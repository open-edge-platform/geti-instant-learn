# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from api.error_handler import custom_exception_handler
from api.routers import projects_router
from dependencies import SessionDep, get_prompt_service
from domain.db.models import PromptType
from domain.errors import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
    ResourceUpdateConflictError,
)
from domain.services.schemas.annotation import AnnotationSchema, Point, RectangleAnnotation
from domain.services.schemas.base import Pagination
from domain.services.schemas.prompt import PromptsListSchema, TextPromptSchema, VisualPromptSchema

PROJECT_ID = uuid4()
PROMPT_ID_1 = uuid4()
PROMPT_ID_2 = uuid4()
FRAME_ID = uuid4()
LABEL_ID = uuid4()


def make_text_prompt_schema(prompt_id: UUID, content: str = "test prompt") -> TextPromptSchema:
    return TextPromptSchema(
        id=prompt_id,
        type=PromptType.TEXT,
        content=content,
    )


def make_visual_prompt_schema(prompt_id: UUID, frame_id: UUID, label_id: UUID | None = None) -> VisualPromptSchema:
    return VisualPromptSchema(
        id=prompt_id,
        type=PromptType.VISUAL,
        frame_id=frame_id,
        annotations=[
            AnnotationSchema(
                config=RectangleAnnotation(type="rectangle", points=[Point(x=0.1, y=0.1), Point(x=0.5, y=0.5)]),
                label_id=label_id,
            )
        ],
    )


@pytest.fixture
def app():
    from api.endpoints import prompts as _  # noqa: F401

    app = FastAPI()
    app.include_router(projects_router, prefix="/api/v1")
    app.dependency_overrides[SessionDep] = lambda: object()

    app.add_exception_handler(Exception, custom_exception_handler)
    app.add_exception_handler(RequestValidationError, custom_exception_handler)

    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    "behavior,expected_status,expected_count",
    [
        ("some", 200, 2),
        ("empty", 200, 0),
        ("notfound", 404, None),
        ("error", 500, None),
    ],
)
def test_get_all_prompts(client, behavior, expected_status, expected_count):
    class FakeService:
        def __init__(self, session, prompt_repository, project_repository, frame_repository, label_repository):
            pass

        def list_prompts(self, project_id: UUID, offset: int = 0, limit: int = 10):
            assert project_id == PROJECT_ID
            if behavior == "some":
                return PromptsListSchema(
                    prompts=[
                        make_text_prompt_schema(PROMPT_ID_1, "find red car"),
                        make_visual_prompt_schema(PROMPT_ID_2, FRAME_ID, LABEL_ID),
                    ],
                    pagination=Pagination(count=2, total=2, offset=offset, limit=limit),
                )
            if behavior == "empty":
                return PromptsListSchema(
                    prompts=[],
                    pagination=Pagination(count=0, total=0, offset=offset, limit=limit),
                )
            if behavior == "notfound":
                raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_prompt_service] = lambda: FakeService(None, None, None, None, None)

    resp = client.get(f"/api/v1/projects/{PROJECT_ID}/prompts")
    assert resp.status_code == expected_status
    if behavior in ("some", "empty"):
        data = resp.json()
        assert "prompts" in data
        assert "pagination" in data
        assert len(data["prompts"]) == expected_count
        if behavior == "some":
            assert data["prompts"][0]["type"] == "TEXT"
            assert data["prompts"][0]["content"] == "find red car"
            assert data["prompts"][1]["type"] == "VISUAL"
            assert "frame_id" in data["prompts"][1]
            assert "annotations" in data["prompts"][1]
    else:
        assert "detail" in resp.json()


@pytest.mark.parametrize(
    "behavior,expected_status",
    [
        ("text_prompt", 200),
        ("visual_prompt", 200),
        ("notfound", 404),
        ("error", 500),
    ],
)
def test_get_prompt(client, behavior, expected_status):
    class FakeService:
        def __init__(self, session, prompt_repository, project_repository, frame_repository, label_repository):
            pass

        def get_prompt(self, project_id: UUID, prompt_id: UUID):
            assert project_id == PROJECT_ID
            assert prompt_id == PROMPT_ID_1
            if behavior == "text_prompt":
                return make_text_prompt_schema(PROMPT_ID_1, "search query")
            if behavior == "visual_prompt":
                return make_visual_prompt_schema(PROMPT_ID_1, FRAME_ID, LABEL_ID)
            if behavior == "notfound":
                raise ResourceNotFoundError(ResourceType.PROMPT, str(prompt_id))
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_prompt_service] = lambda: FakeService(None, None, None, None, None)

    resp = client.get(f"/api/v1/projects/{PROJECT_ID}/prompts/{PROMPT_ID_1}")
    assert resp.status_code == expected_status
    if behavior == "text_prompt":
        data = resp.json()
        assert data["id"] == str(PROMPT_ID_1)
        assert data["type"] == "TEXT"
        assert data["content"] == "search query"
    elif behavior == "visual_prompt":
        data = resp.json()
        assert data["id"] == str(PROMPT_ID_1)
        assert data["type"] == "VISUAL"
        assert data["frame_id"] == str(FRAME_ID)
        assert len(data["annotations"]) == 1
    else:
        assert "detail" in resp.json()


@pytest.mark.parametrize(
    "behavior,expected_status",
    [
        ("text_success", 201),
        ("visual_success", 201),
        ("text_exists", 409),
        ("frame_notfound", 404),
        ("label_notfound", 404),
        ("project_notfound", 404),
        ("error", 500),
    ],
)
def test_create_prompt(client, behavior, expected_status):  # noqa: C901
    CREATED_ID = uuid4()

    class FakeService:
        def __init__(self, session, prompt_repository, project_repository, frame_repository, label_repository):
            pass

        def create_prompt(self, project_id: UUID, create_data):
            assert project_id == PROJECT_ID
            if behavior == "text_success":
                assert create_data.type == PromptType.TEXT
                return make_text_prompt_schema(CREATED_ID, create_data.content)
            if behavior == "visual_success":
                assert create_data.type == PromptType.VISUAL
                return make_visual_prompt_schema(CREATED_ID, create_data.frame_id, LABEL_ID)
            if behavior == "text_exists":
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.PROMPT,
                    field="type",
                    message="A text prompt already exists for this project.",
                )
            if behavior == "frame_notfound":
                raise ResourceNotFoundError(ResourceType.FRAME, str(create_data.frame_id))
            if behavior == "label_notfound":
                raise ResourceNotFoundError(ResourceType.LABEL, str(LABEL_ID))
            if behavior == "project_notfound":
                raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_prompt_service] = lambda: FakeService(None, None, None, None, None)

    if behavior in ("text_success", "text_exists"):
        payload = {
            "id": str(CREATED_ID),
            "type": "TEXT",
            "content": "find red car",
        }
    else:
        payload = {
            "id": str(CREATED_ID),
            "type": "VISUAL",
            "frame_id": str(FRAME_ID),
            "annotations": [
                {
                    "config": {
                        "type": "rectangle",
                        "points": [{"x": 0.1, "y": 0.1}, {"x": 0.5, "y": 0.5}],
                    },
                    "label_id": str(LABEL_ID) if behavior != "label_notfound" else str(uuid4()),
                }
            ],
        }

    resp = client.post(f"/api/v1/projects/{PROJECT_ID}/prompts", json=payload)
    assert resp.status_code == expected_status
    if behavior == "text_success":
        data = resp.json()
        assert data["id"] == str(CREATED_ID)
        assert data["type"] == "TEXT"
        assert data["content"] == "find red car"
    elif behavior == "visual_success":
        data = resp.json()
        assert data["id"] == str(CREATED_ID)
        assert data["type"] == "VISUAL"
        assert data["frame_id"] == str(FRAME_ID)
        assert len(data["annotations"]) == 1
    else:
        assert "detail" in resp.json()


@pytest.mark.parametrize(
    "behavior,expected_status",
    [
        ("text_success", 200),
        ("visual_success", 200),
        ("type_conflict", 400),
        ("frame_notfound", 404),
        ("label_notfound", 404),
        ("prompt_notfound", 404),
        ("error", 500),
    ],
)
def test_update_prompt(client, behavior, expected_status):  # noqa: C901
    class FakeService:
        def __init__(self, session, prompt_repository, project_repository, frame_repository, label_repository):
            pass

        def update_prompt(self, project_id: UUID, prompt_id: UUID, update_data):
            assert project_id == PROJECT_ID
            assert prompt_id == PROMPT_ID_1
            if behavior == "text_success":
                assert update_data.type == PromptType.TEXT
                return make_text_prompt_schema(PROMPT_ID_1, update_data.content or "updated content")
            if behavior == "visual_success":
                assert update_data.type == PromptType.VISUAL
                return make_visual_prompt_schema(PROMPT_ID_1, update_data.frame_id or FRAME_ID, LABEL_ID)
            if behavior == "type_conflict":
                raise ResourceUpdateConflictError(
                    resource_type=ResourceType.PROMPT,
                    resource_id=str(prompt_id),
                    field="type",
                    message="Cannot change prompt type.",
                )
            if behavior == "frame_notfound":
                raise ResourceNotFoundError(ResourceType.FRAME, str(update_data.frame_id))
            if behavior == "label_notfound":
                raise ResourceNotFoundError(ResourceType.LABEL, str(LABEL_ID))
            if behavior == "prompt_notfound":
                raise ResourceNotFoundError(ResourceType.PROMPT, str(prompt_id))
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_prompt_service] = lambda: FakeService(None, None, None, None, None)

    if behavior in ("text_success", "type_conflict"):
        payload = {
            "type": "TEXT",
            "content": "updated text prompt",
        }
    else:
        new_frame_id = uuid4() if behavior == "frame_notfound" else FRAME_ID
        payload = {
            "type": "VISUAL",
            "frame_id": str(new_frame_id),
            "annotations": [
                {
                    "config": {
                        "type": "rectangle",
                        "points": [{"x": 0.2, "y": 0.2}, {"x": 0.7, "y": 0.7}],
                    },
                    "label_id": str(LABEL_ID),
                }
            ],
        }

    resp = client.put(f"/api/v1/projects/{PROJECT_ID}/prompts/{PROMPT_ID_1}", json=payload)
    assert resp.status_code == expected_status
    if behavior == "text_success":
        data = resp.json()
        assert data["id"] == str(PROMPT_ID_1)
        assert data["type"] == "TEXT"
        assert data["content"] == "updated text prompt"
    elif behavior == "visual_success":
        data = resp.json()
        assert data["id"] == str(PROMPT_ID_1)
        assert data["type"] == "VISUAL"
        assert "frame_id" in data
        assert "annotations" in data
    else:
        assert "detail" in resp.json()


@pytest.mark.parametrize(
    "behavior,expected_status",
    [
        ("success", 204),
        ("missing", 404),
        ("error", 500),
    ],
)
def test_delete_prompt(client, behavior, expected_status):
    class FakeService:
        def __init__(self, session, prompt_repository, project_repository, frame_repository, label_repository):
            pass

        def delete_prompt(self, project_id: UUID, prompt_id: UUID):
            assert project_id == PROJECT_ID
            assert prompt_id == PROMPT_ID_2
            if behavior == "success":
                return
            if behavior == "missing":
                raise ResourceNotFoundError(ResourceType.PROMPT, str(prompt_id))
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_prompt_service] = lambda: FakeService(None, None, None, None, None)

    resp = client.delete(f"/api/v1/projects/{PROJECT_ID}/prompts/{PROMPT_ID_2}")
    assert resp.status_code == expected_status
    if expected_status == 204:
        assert resp.text == ""
    else:
        assert "detail" in resp.json()


def test_get_prompts_pagination(client):
    class FakeService:
        def __init__(self, session, prompt_repository, project_repository, frame_repository, label_repository):
            pass

        def list_prompts(self, project_id: UUID, offset: int = 0, limit: int = 10):
            assert offset == 5
            assert limit == 15
            return PromptsListSchema(
                prompts=[make_text_prompt_schema(PROMPT_ID_1)],
                pagination=Pagination(count=1, total=20, offset=offset, limit=limit),
            )

    client.app.dependency_overrides[get_prompt_service] = lambda: FakeService(None, None, None, None, None)

    resp = client.get(f"/api/v1/projects/{PROJECT_ID}/prompts?offset=5&limit=15")
    assert resp.status_code == 200
    data = resp.json()
    assert data["pagination"]["offset"] == 5
    assert data["pagination"]["limit"] == 15
    assert data["pagination"]["total"] == 20


def test_create_prompt_invalid_annotation_type(client):
    class FakeService:
        def __init__(self, session, prompt_repository, project_repository, frame_repository, label_repository):
            pass

    client.app.dependency_overrides[get_prompt_service] = lambda: FakeService(None, None, None, None, None)

    payload = {
        "id": str(uuid4()),
        "type": "VISUAL",
        "frame_id": str(FRAME_ID),
        "annotations": [
            {
                "config": {"type": "invalid_type", "points": [{"x": 0.5, "y": 0.5}]},
                "label_id": str(LABEL_ID),
            }
        ],
    }

    resp = client.post(f"/api/v1/projects/{PROJECT_ID}/prompts", json=payload)
    assert resp.status_code == 400
    assert "detail" in resp.json()
