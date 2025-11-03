# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from core.components.schemas.reader import SourceType, WebCamConfig
from dependencies import SessionDep, get_config_dispatcher, get_source_service
from exceptions.custom_errors import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
    ResourceUpdateConflictError,
)
from exceptions.handler import custom_exception_handler
from routers import projects_router
from services.schemas.source import SourceSchema, SourcesListSchema

PROJECT_ID = uuid4()
SOURCE_ID_1 = uuid4()
SOURCE_ID_2 = uuid4()


def make_source_schema(
    source_id: UUID,
    device_id: int,
    connected: bool = False,
) -> SourceSchema:
    return SourceSchema(
        id=source_id,
        connected=connected,
        config=WebCamConfig(source_type=SourceType.WEBCAM, device_id=device_id),
    )


@pytest.fixture
def app():
    from rest.endpoints import sources as _  # noqa: F401

    app = FastAPI()
    app.include_router(projects_router, prefix="/api/v1")
    app.dependency_overrides[SessionDep] = lambda: object()

    class DummyDispatcher:
        def dispatch(self, event):
            pass

    app.dependency_overrides[get_config_dispatcher] = lambda: DummyDispatcher()

    app.add_exception_handler(Exception, custom_exception_handler)
    app.add_exception_handler(RequestValidationError, custom_exception_handler)

    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    "behavior,expected_status,expected_len",
    [
        ("some", 200, 2),
        ("notfound", 404, None),
        ("error", 500, None),
    ],
)
def test_get_sources(client, behavior, expected_status, expected_len):
    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def list_sources(self, project_id: UUID):
            assert project_id == PROJECT_ID
            if behavior == "some":
                return SourcesListSchema(
                    sources=[
                        make_source_schema(SOURCE_ID_1, 0, True),
                        make_source_schema(SOURCE_ID_2, 1, False),
                    ]
                )
            if behavior == "notfound":
                raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_source_service] = lambda: FakeService(None, None)

    resp = client.get(f"/api/v1/projects/{PROJECT_ID}/sources")
    assert resp.status_code == expected_status
    if behavior == "some":
        data = resp.json()
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) == expected_len
        ids = {s["id"] for s in data["sources"]}
        assert ids == {str(SOURCE_ID_1), str(SOURCE_ID_2)}
        first = data["sources"][0]
        assert "config" in first
        assert first["config"]["source_type"] == "webcam"
        assert "device_id" in first["config"]
    else:
        assert "detail" in resp.json()


@pytest.mark.parametrize(
    "behavior,expected_status",
    [
        ("success", 201),
        ("conflict_type", 409),
        ("conflict_connected", 409),
        ("notfound", 404),
        ("error", 500),
    ],
)
def test_create_source(client, behavior, expected_status):
    CREATED_ID = uuid4()

    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def create_source(self, project_id: UUID, create_data):
            assert project_id == PROJECT_ID
            assert create_data.config.source_type == SourceType.WEBCAM
            if behavior == "success":
                return make_source_schema(CREATED_ID, create_data.config.device_id, create_data.connected)
            if behavior == "conflict_type":
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.SOURCE,
                    resource_value="source_type",
                    field="source_type",
                    message="A source with this type already exists in the project.",
                )
            if behavior == "conflict_connected":
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.SOURCE,
                    resource_value="connected",
                    field="connected",
                    message="Only one source can be connected per project at a time.",
                )
            if behavior == "notfound":
                raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_source_service] = lambda: FakeService(None, None)

    payload = {
        "id": str(CREATED_ID),
        "connected": True,
        "config": {"source_type": "webcam", "device_id": 3},
    }
    resp = client.post(f"/api/v1/projects/{PROJECT_ID}/sources", json=payload)
    assert resp.status_code == expected_status
    if behavior == "success":
        data = resp.json()
        assert data["id"] == str(CREATED_ID)
        assert data["connected"] is True
        assert data["config"]["source_type"] == "webcam"
        assert data["config"]["device_id"] == 3
    else:
        assert "detail" in resp.json()


@pytest.mark.parametrize(
    "behavior,expected_status",
    [
        ("success", 200),
        ("conflict", 400),
        ("notfound", 404),
        ("error", 500),
    ],
)
def test_update_source(client, behavior, expected_status):
    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def update_source(self, project_id: UUID, source_id: UUID, update_data):
            assert project_id == PROJECT_ID
            assert source_id == SOURCE_ID_1
            assert update_data.config.source_type == SourceType.WEBCAM
            if behavior == "success":
                return make_source_schema(source_id, update_data.config.device_id, update_data.connected)
            if behavior == "conflict":
                raise ResourceUpdateConflictError(
                    resource_type=ResourceType.SOURCE,
                    resource_id=str(source_id),
                    field="source_type",
                    message="Cannot change source type after creation.",
                )
            if behavior == "notfound":
                raise ResourceNotFoundError(ResourceType.SOURCE, str(source_id))
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_source_service] = lambda: FakeService(None, None)

    payload = {
        "connected": False,
        "config": {"source_type": "webcam", "device_id": 7},
    }
    resp = client.put(f"/api/v1/projects/{PROJECT_ID}/sources/{SOURCE_ID_1}", json=payload)
    assert resp.status_code == expected_status
    if behavior == "success":
        data = resp.json()
        assert data["id"] == str(SOURCE_ID_1)
        assert data["config"]["device_id"] == 7
        assert data["config"]["source_type"] == "webcam"
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
def test_delete_source(client, behavior, expected_status):
    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def delete_source(self, project_id: UUID, source_id: UUID):
            assert project_id == PROJECT_ID
            assert source_id == SOURCE_ID_2
            if behavior == "success":
                return
            if behavior == "missing":
                raise ResourceNotFoundError(ResourceType.SOURCE, str(source_id))
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_source_service] = lambda: FakeService(None, None)

    resp = client.delete(f"/api/v1/projects/{PROJECT_ID}/sources/{SOURCE_ID_2}")
    assert resp.status_code == expected_status
    if expected_status == 204:
        assert resp.text == ""
    else:
        assert "detail" in resp.json()
