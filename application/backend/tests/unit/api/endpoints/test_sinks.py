# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from api.error_handler import custom_exception_handler
from api.routers import projects_router
from dependencies import SessionDep, get_config_dispatcher, get_sink_connection_validator, get_sink_service
from domain.errors import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
    ResourceUpdateConflictError,
)
from domain.services.schemas.base import Pagination
from domain.services.schemas.sink import SinkSchema, SinksListSchema
from domain.services.schemas.writer import MqttConfig, WriterType

PROJECT_ID = uuid4()
SINK_ID_1 = uuid4()
SINK_ID_2 = uuid4()


@pytest.fixture
def app():
    from api.endpoints import sources as _  # noqa: F401

    app = FastAPI()
    app.include_router(projects_router, prefix="/api/v1")
    app.dependency_overrides[SessionDep] = lambda: object()

    class DummyDispatcher:
        def dispatch(self, event):
            pass

    app.dependency_overrides[get_config_dispatcher] = lambda: DummyDispatcher()
    validator = Mock()
    validator.validate.return_value = None
    app.dependency_overrides[get_sink_connection_validator] = lambda: validator

    app.add_exception_handler(Exception, custom_exception_handler)
    app.add_exception_handler(RequestValidationError, custom_exception_handler)

    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def make_sink_schema(
    sink_id: UUID,
    active: bool = False,
) -> SinkSchema:
    return SinkSchema(
        id=sink_id,
        active=active,
        config=MqttConfig(sink_type=WriterType.MQTT),
    )


@pytest.mark.parametrize(
    "behavior,expected_status,expected_len",
    [
        ("some", 200, 2),
        ("notfound", 404, None),
        ("error", 500, None),
    ],
)
def test_get_sinks(client, behavior, expected_status, expected_len):
    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        @staticmethod
        def list_sinks(project_id: UUID, offset: int = 0, limit: int = 20):
            assert project_id == PROJECT_ID
            if behavior == "some":
                sinks = [
                    make_sink_schema(sink_id=SINK_ID_1, active=True),
                    make_sink_schema(sink_id=SINK_ID_2, active=False),
                ]
                return SinksListSchema(
                    sinks=sinks, pagination=Pagination(count=len(sinks), total=len(sinks), offset=offset, limit=limit)
                )
            if behavior == "notfound":
                raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_sink_service] = lambda: FakeService(None, None)

    resp = client.get(f"/api/v1/projects/{PROJECT_ID}/sinks")
    assert resp.status_code == expected_status
    if behavior == "some":
        data = resp.json()
        assert isinstance(data["sinks"], list)
        assert len(data["sinks"]) == expected_len
        ids = {s["id"] for s in data["sinks"]}
        assert ids == {str(SINK_ID_1), str(SINK_ID_2)}
        first = data["sinks"][0]
        assert "config" in first
        assert first["config"]["sink_type"] == "mqtt"
        assert "pagination" in data
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
def test_create_sink(client, behavior, expected_status):
    CREATED_ID = uuid4()

    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def create_sink(self, project_id: UUID, create_data):
            assert project_id == PROJECT_ID
            assert create_data.config.sink_type == WriterType.MQTT
            if behavior == "success":
                return make_sink_schema(sink_id=CREATED_ID, active=create_data.active)
            if behavior == "conflict_type":
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.SINK,
                    resource_value="sink_type",
                    field="sink_type",
                    message="A sink with this type already exists in the project.",
                )
            if behavior == "conflict_connected":
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.SINK,
                    resource_value="active",
                    field="active",
                    message="Only one sink can be active per project at a time.",
                )
            if behavior == "notfound":
                raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_sink_service] = lambda: FakeService(None, None)

    payload = {
        "id": str(CREATED_ID),
        "active": True,
        "config": {"sink_type": "mqtt"},
    }
    resp = client.post(f"/api/v1/projects/{PROJECT_ID}/sinks", json=payload)
    assert resp.status_code == expected_status
    if behavior == "success":
        data = resp.json()
        assert data["id"] == str(CREATED_ID)
        assert data["active"] is True
        assert data["config"]["sink_type"] == WriterType.MQTT
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
def test_update_sink(client, behavior, expected_status):
    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def update_sink(self, project_id: UUID, sink_id: UUID, update_data):
            assert project_id == PROJECT_ID
            assert sink_id == SINK_ID_1
            assert update_data.config.sink_type == WriterType.MQTT
            if behavior == "success":
                return make_sink_schema(sink_id=sink_id, active=update_data.active)
            if behavior == "conflict":
                raise ResourceUpdateConflictError(
                    resource_type=ResourceType.SINK,
                    resource_id=str(sink_id),
                    field="sink_type",
                    message="Cannot change sink type after creation.",
                )
            if behavior == "notfound":
                raise ResourceNotFoundError(ResourceType.SINK, str(sink_id))
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_sink_service] = lambda: FakeService(None, None)

    payload = {
        "active": False,
        "config": {"sink_type": "mqtt"},
    }
    resp = client.put(f"/api/v1/projects/{PROJECT_ID}/sinks/{SINK_ID_1}", json=payload)
    assert resp.status_code == expected_status
    if behavior == "success":
        data = resp.json()
        assert data["id"] == str(SINK_ID_1)
        assert data["config"]["sink_type"] == WriterType.MQTT
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

        def delete_sink(self, project_id: UUID, sink_id: UUID):
            assert project_id == PROJECT_ID
            assert sink_id == SINK_ID_1
            if behavior == "success":
                return
            if behavior == "missing":
                raise ResourceNotFoundError(ResourceType.SINK, str(sink_id))
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_sink_service] = lambda: FakeService(None, None)

    resp = client.delete(f"/api/v1/projects/{PROJECT_ID}/sinks/{SINK_ID_1}")
    assert resp.status_code == expected_status
    if expected_status == 204:
        assert resp.text == ""
    else:
        assert "detail" in resp.json()
