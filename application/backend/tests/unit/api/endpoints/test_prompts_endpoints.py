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
from domain.services.schemas.base import Pagination
from domain.services.schemas.prompt import (
    PromptsListSchema,
    TextPromptSchema,
    VisualPromptListItemSchema,
    VisualPromptSchema,
)

PROJECT_ID = uuid4()
PROJECT_ID_STR = str(PROJECT_ID)
PROMPT_ID = uuid4()
PROMPT_ID_STR = str(PROMPT_ID)
SECOND_PROMPT_ID = uuid4()
SECOND_PROMPT_ID_STR = str(SECOND_PROMPT_ID)
FRAME_ID = uuid4()
FRAME_ID_STR = str(FRAME_ID)
LABEL_ID = uuid4()
LABEL_ID_STR = str(LABEL_ID)


def assert_text_prompt_schema(data: dict, prompt_id: str, content: str):
    assert data["id"] == prompt_id
    assert data["type"] == "TEXT"
    assert data["content"] == content


def assert_visual_prompt_schema(data: dict, prompt_id: str, frame_id: str):
    assert data["id"] == prompt_id
    assert data["type"] == "VISUAL"
    assert data["frame_id"] == frame_id
    assert "annotations" in data
    assert isinstance(data["annotations"], list)


@pytest.fixture
def app():
    from api.endpoints import prompts as _  # noqa: F401

    app = FastAPI()
    app.include_router(projects_router, prefix="/api/v1")
    app.dependency_overrides[SessionDep] = lambda: object()

    class DummyDispatcher:
        def dispatch(self, event):
            pass

    from dependencies import get_config_dispatcher

    app.dependency_overrides[get_config_dispatcher] = lambda: DummyDispatcher()

    app.add_exception_handler(Exception, custom_exception_handler)
    app.add_exception_handler(RequestValidationError, custom_exception_handler)

    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


# GET /projects/{project_id}/prompts
@pytest.mark.parametrize(
    "behavior,expected_status,expected_count",
    [
        ("no_prompts", 200, 0),
        ("some_prompts", 200, 2),
        ("error", 500, None),
    ],
)
def test_get_all_prompts(client, behavior, expected_status, expected_count):
    class FakeService:
        def __init__(self, session, prompt_repository, project_repository, frame_repository, label_repository):
            pass

        def list_prompts(self, project_id: UUID, offset=0, limit=10):
            assert project_id == PROJECT_ID
            if behavior == "no_prompts":
                return PromptsListSchema(
                    prompts=[], pagination=Pagination(count=0, total=0, offset=offset, limit=limit)
                )
            if behavior == "some_prompts":
                prompts = [
                    TextPromptSchema(id=PROMPT_ID, type=PromptType.TEXT, content="find red car"),
                    VisualPromptListItemSchema(
                        id=SECOND_PROMPT_ID,
                        type=PromptType.VISUAL,
                        frame_id=FRAME_ID,
                        annotations=[],
                        thumbnail="data:image/jpeg;base64,fake",
                    ),
                ]
                return PromptsListSchema(
                    prompts=prompts, pagination=Pagination(count=2, total=2, offset=offset, limit=limit)
                )
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_prompt_service] = lambda: FakeService(None, None, None, None, None)

    resp = client.get(f"/api/v1/projects/{PROJECT_ID_STR}/prompts")
    assert resp.status_code == expected_status

    if behavior == "error":
        assert "detail" in resp.json()
        return

    data = resp.json()
    prompts_list = data["prompts"]
    assert len(prompts_list) == expected_count
    assert "pagination" in data
    pagination = data["pagination"]
    assert pagination["count"] == expected_count
    assert pagination["offset"] == 0
    assert pagination["limit"] == 10

    if behavior == "some_prompts":
        assert prompts_list[0]["type"] == "TEXT"
        assert prompts_list[1]["type"] == "VISUAL"
        # Thumbnail should be present in list response
        assert "thumbnail" in prompts_list[1]
        assert prompts_list[1]["thumbnail"] == "data:image/jpeg;base64,fake"


def test_get_all_prompts_with_pagination(client):
    class FakeService:
        def __init__(self, session, prompt_repository, project_repository, frame_repository, label_repository):
            pass

        def list_prompts(self, project_id: UUID, offset=0, limit=10):
            assert offset == 5
            assert limit == 15
            prompts = [TextPromptSchema(id=PROMPT_ID, type=PromptType.TEXT, content="test")]
            return PromptsListSchema(prompts=prompts, pagination=Pagination(count=1, total=20, offset=5, limit=15))

    client.app.dependency_overrides[get_prompt_service] = lambda: FakeService(None, None, None, None, None)

    resp = client.get(f"/api/v1/projects/{PROJECT_ID_STR}/prompts?offset=5&limit=15")
    assert resp.status_code == 200
    data = resp.json()
    assert data["pagination"]["offset"] == 5
    assert data["pagination"]["limit"] == 15
    assert data["pagination"]["total"] == 20


# GET /projects/{project_id}/prompts/{prompt_id}
@pytest.mark.parametrize(
    "prompt_type,behavior,expected_status",
    [
        ("text", "success", 200),
        ("visual", "success", 200),
        ("text", "notfound", 404),
        ("text", "error", 500),
    ],
)
def test_get_prompt(client, prompt_type, behavior, expected_status):
    class FakeService:
        def __init__(self, session, prompt_repository, project_repository, frame_repository, label_repository):
            pass

        def get_prompt(self, project_id: UUID, prompt_id: UUID):
            assert project_id == PROJECT_ID
            assert prompt_id == PROMPT_ID
            if behavior == "success":
                if prompt_type == "text":
                    return TextPromptSchema(id=PROMPT_ID, type=PromptType.TEXT, content="find red car")
                return VisualPromptSchema(
                    id=PROMPT_ID,
                    type=PromptType.VISUAL,
                    frame_id=FRAME_ID,
                    annotations=[],
                )
            if behavior == "notfound":
                raise ResourceNotFoundError(resource_type=ResourceType.PROMPT, resource_id=str(prompt_id))
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_prompt_service] = lambda: FakeService(None, None, None, None, None)

    resp = client.get(f"/api/v1/projects/{PROJECT_ID_STR}/prompts/{PROMPT_ID_STR}")
    assert resp.status_code == expected_status

    if behavior == "success":
        data = resp.json()
        if prompt_type == "text":
            assert_text_prompt_schema(data, PROMPT_ID_STR, "find red car")
        else:
            assert_visual_prompt_schema(data, PROMPT_ID_STR, FRAME_ID_STR)
            # Thumbnail should NOT be present in detail response
            assert data.get("thumbnail") is None
    else:
        assert "detail" in resp.json()


# POST /projects/{project_id}/prompts
@pytest.mark.parametrize(
    "behavior,expected_status,expect_location",
    [
        ("success", 201, False),
        ("text_duplicate", 409, False),
        ("error", 500, False),
    ],
)
def test_create_text_prompt(client, behavior, expected_status, expect_location):
    class FakeService:
        def __init__(self, session, prompt_repository, project_repository, frame_repository, label_repository):
            pass

        def create_prompt(self, project_id: UUID, create_data):
            assert project_id == PROJECT_ID
            assert create_data.type == PromptType.TEXT
            assert create_data.content == "find red car"

            if behavior == "success":
                return TextPromptSchema(id=PROMPT_ID, type=PromptType.TEXT, content="find red car")
            if behavior == "text_duplicate":
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.PROMPT,
                    field="type",
                    message="A text prompt already exists for this project.",
                )
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_prompt_service] = lambda: FakeService(None, None, None, None, None)

    payload = {"type": "TEXT", "content": "find red car"}
    resp = client.post(f"/api/v1/projects/{PROJECT_ID_STR}/prompts", json=payload)

    assert resp.status_code == expected_status
    if behavior == "success":
        response_data = resp.json()
        assert_text_prompt_schema(response_data, PROMPT_ID_STR, "find red car")
    else:
        assert "detail" in resp.json()


@pytest.mark.parametrize(
    "behavior,expected_status,expect_location",
    [
        ("success", 201, False),
        ("frame_not_found", 404, False),
        ("label_not_found", 404, False),
        ("frame_duplicate", 409, False),
        ("error", 500, False),
    ],
)
def test_create_visual_prompt(client, behavior, expected_status, expect_location):
    class FakeService:
        def __init__(self, session, prompt_repository, project_repository, frame_repository, label_repository):
            pass

        def create_prompt(self, project_id: UUID, create_data):
            assert project_id == PROJECT_ID
            assert create_data.type == PromptType.VISUAL
            assert create_data.frame_id == FRAME_ID
            assert len(create_data.annotations) == 1

            if behavior == "success":
                return VisualPromptSchema(
                    id=PROMPT_ID,
                    type=PromptType.VISUAL,
                    frame_id=FRAME_ID,
                    annotations=create_data.annotations,
                )
            if behavior == "frame_not_found":
                raise ResourceNotFoundError(
                    resource_type=ResourceType.FRAME,
                    resource_id=str(FRAME_ID),
                    message=f"Frame {FRAME_ID} does not exist",
                )
            if behavior == "label_not_found":
                raise ResourceNotFoundError(
                    resource_type=ResourceType.LABEL,
                    resource_id=str(LABEL_ID),
                    message=f"Label {LABEL_ID} does not exist",
                )
            if behavior == "frame_duplicate":
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.PROMPT,
                    field="frame_id",
                    message=f"Frame {FRAME_ID} is already used by another prompt.",
                )
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_prompt_service] = lambda: FakeService(None, None, None, None, None)

    payload = {
        "type": "VISUAL",
        "frame_id": FRAME_ID_STR,
        "annotations": [
            {
                "config": {"type": "rectangle", "points": [{"x": 0.1, "y": 0.1}, {"x": 0.5, "y": 0.5}]},
                "label_id": LABEL_ID_STR,
            }
        ],
    }
    resp = client.post(f"/api/v1/projects/{PROJECT_ID_STR}/prompts", json=payload)

    assert resp.status_code == expected_status
    if behavior == "success":
        response_data = resp.json()
        assert_visual_prompt_schema(response_data, PROMPT_ID_STR, FRAME_ID_STR)
        # Thumbnail should NOT be present in create response
        assert response_data.get("thumbnail") is None
    else:
        assert "detail" in resp.json()


# PUT /projects/{project_id}/prompts/{prompt_id}
@pytest.mark.parametrize(
    "behavior,expected_status",
    [
        ("success", 200),
        ("notfound", 404),
        ("type_conflict", 400),
        ("error", 500),
    ],
)
def test_update_text_prompt(client, behavior, expected_status):
    class FakeService:
        def __init__(self, session, prompt_repository, project_repository, frame_repository, label_repository):
            pass

        def update_prompt(self, project_id: UUID, prompt_id: UUID, update_data):
            assert project_id == PROJECT_ID
            assert prompt_id == PROMPT_ID
            assert update_data.type == PromptType.TEXT

            if behavior == "success":
                return TextPromptSchema(id=PROMPT_ID, type=PromptType.TEXT, content="updated content")
            if behavior == "notfound":
                raise ResourceNotFoundError(resource_type=ResourceType.PROMPT, resource_id=str(prompt_id))
            if behavior == "type_conflict":
                raise ResourceUpdateConflictError(
                    resource_type=ResourceType.PROMPT,
                    resource_id=str(prompt_id),
                    field="type",
                    message="Cannot change prompt type",
                )
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_prompt_service] = lambda: FakeService(None, None, None, None, None)

    payload = {"type": "TEXT", "content": "updated content"}
    resp = client.put(f"/api/v1/projects/{PROJECT_ID_STR}/prompts/{PROMPT_ID_STR}", json=payload)

    assert resp.status_code == expected_status
    if behavior == "success":
        assert_text_prompt_schema(resp.json(), PROMPT_ID_STR, "updated content")
    else:
        assert "detail" in resp.json()


@pytest.mark.parametrize(
    "behavior,expected_status",
    [
        ("success", 200),
        ("notfound", 404),
        ("frame_not_found", 404),
        ("error", 500),
    ],
)
def test_update_visual_prompt(client, behavior, expected_status):
    NEW_FRAME_ID = uuid4()
    NEW_FRAME_ID_STR = str(NEW_FRAME_ID)

    class FakeService:
        def __init__(self, session, prompt_repository, project_repository, frame_repository, label_repository):
            pass

        def update_prompt(self, project_id: UUID, prompt_id: UUID, update_data):
            assert project_id == PROJECT_ID
            assert prompt_id == PROMPT_ID
            assert update_data.type == PromptType.VISUAL

            if behavior == "success":
                return VisualPromptSchema(
                    id=PROMPT_ID,
                    type=PromptType.VISUAL,
                    frame_id=NEW_FRAME_ID,
                    annotations=update_data.annotations or [],
                )
            if behavior == "notfound":
                raise ResourceNotFoundError(resource_type=ResourceType.PROMPT, resource_id=str(prompt_id))
            if behavior == "frame_not_found":
                raise ResourceNotFoundError(
                    resource_type=ResourceType.FRAME,
                    resource_id=str(NEW_FRAME_ID),
                )
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_prompt_service] = lambda: FakeService(None, None, None, None, None)

    payload = {
        "type": "VISUAL",
        "frame_id": NEW_FRAME_ID_STR,
        "annotations": [
            {
                "config": {"type": "rectangle", "points": [{"x": 0.2, "y": 0.2}, {"x": 0.7, "y": 0.7}]},
                "label_id": LABEL_ID_STR,
            }
        ],
    }
    resp = client.put(f"/api/v1/projects/{PROJECT_ID_STR}/prompts/{PROMPT_ID_STR}", json=payload)

    assert resp.status_code == expected_status
    if behavior == "success":
        data = resp.json()
        assert_visual_prompt_schema(data, PROMPT_ID_STR, NEW_FRAME_ID_STR)
    else:
        assert "detail" in resp.json()


# DELETE /projects/{project_id}/prompts/{prompt_id}
@pytest.mark.parametrize(
    "behavior,expected_status",
    [
        ("success", 204),
        ("notfound", 404),
        ("error", 500),
    ],
)
def test_delete_prompt(client, behavior, expected_status):
    class FakeService:
        def __init__(self, session, prompt_repository, project_repository, frame_repository, label_repository):
            pass

        def delete_prompt(self, project_id: UUID, prompt_id: UUID):
            assert project_id == PROJECT_ID
            assert prompt_id == PROMPT_ID

            if behavior == "success":
                return
            if behavior == "notfound":
                raise ResourceNotFoundError(resource_type=ResourceType.PROMPT, resource_id=str(prompt_id))
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_prompt_service] = lambda: FakeService(None, None, None, None, None)

    resp = client.delete(f"/api/v1/projects/{PROJECT_ID_STR}/prompts/{PROMPT_ID_STR}")
    assert resp.status_code == expected_status

    if expected_status == 204:
        assert resp.text == ""
    else:
        assert "detail" in resp.json()


# Validation tests
def test_create_prompt_validation_error_empty_content(client):
    payload = {"type": "TEXT", "content": ""}
    resp = client.post(f"/api/v1/projects/{PROJECT_ID_STR}/prompts", json=payload)
    assert resp.status_code == 400
    assert "detail" in resp.json()


def test_create_prompt_validation_error_missing_frame_id(client):
    payload = {
        "type": "VISUAL",
        "annotations": [
            {
                "config": {"type": "rectangle", "points": [{"x": 0.1, "y": 0.1}, {"x": 0.5, "y": 0.5}]},
                "label_id": LABEL_ID_STR,
            }
        ],
    }
    resp = client.post(f"/api/v1/projects/{PROJECT_ID_STR}/prompts", json=payload)
    assert resp.status_code == 400


def test_create_prompt_validation_error_empty_annotations(client):
    payload = {"type": "VISUAL", "frame_id": FRAME_ID_STR, "annotations": []}
    resp = client.post(f"/api/v1/projects/{PROJECT_ID_STR}/prompts", json=payload)
    assert resp.status_code == 400


def test_create_prompt_validation_error_invalid_annotation_points(client):
    payload = {
        "type": "VISUAL",
        "frame_id": FRAME_ID_STR,
        "annotations": [
            {
                "config": {
                    "type": "rectangle",
                    "points": [{"x": 0.5, "y": 0.5}, {"x": 0.1, "y": 0.1}],  # invalid order
                },
                "label_id": LABEL_ID_STR,
            }
        ],
    }
    resp = client.post(f"/api/v1/projects/{PROJECT_ID_STR}/prompts", json=payload)
    assert resp.status_code == 400
