# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from api.error_handler import custom_exception_handler
from api.routers import projects_router
from dependencies import SessionDep, get_config_dispatcher, get_pipeline_manager, get_source_service
from domain.errors import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
    ResourceUpdateConflictError,
)
from domain.services.schemas.base import Pagination
from domain.services.schemas.reader import FrameListResponse, FrameMetadata, SourceType, UsbCameraConfig
from domain.services.schemas.source import SourceSchema, SourcesListSchema
from runtime.errors import PipelineNotActiveError, SourceNotSeekableError

PROJECT_ID = uuid4()
SOURCE_ID_1 = uuid4()
SOURCE_ID_2 = uuid4()


def make_source_schema(
    source_id: UUID,
    device_id: int,
    active: bool = False,
) -> SourceSchema:
    return SourceSchema(
        id=source_id,
        active=active,
        config=UsbCameraConfig(source_type=SourceType.USB_CAMERA, device_id=device_id),
    )


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

        def list_sources(self, project_id: UUID, offset: int = 0, limit: int = 20):
            assert project_id == PROJECT_ID
            if behavior == "some":
                sources = [
                    make_source_schema(SOURCE_ID_1, 0, True),
                    make_source_schema(SOURCE_ID_2, 1, False),
                ]
                return SourcesListSchema(
                    sources=sources,
                    pagination=Pagination(count=len(sources), total=len(sources), offset=offset, limit=limit),
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
        assert first["config"]["source_type"] == "usb_camera"
        assert "device_id" in first["config"]
        assert "pagination" in data
    else:
        assert "detail" in resp.json()


@pytest.mark.parametrize(
    "behavior,expected_status",
    [
        ("success", 201),
        ("conflict_type", 409),
        ("conflict_active", 409),
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
            assert create_data.config.source_type == SourceType.USB_CAMERA
            if behavior == "success":
                return make_source_schema(CREATED_ID, create_data.config.device_id, create_data.active)
            if behavior == "conflict_type":
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.SOURCE,
                    resource_value="source_type",
                    field="source_type",
                    message="A source with this type already exists in the project.",
                )
            if behavior == "conflict_active":
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.SOURCE,
                    resource_value="active",
                    field="active",
                    message="Only one source can be active per project at a time.",
                )
            if behavior == "notfound":
                raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_source_service] = lambda: FakeService(None, None)

    payload = {
        "id": str(CREATED_ID),
        "active": True,
        "config": {"source_type": "usb_camera", "device_id": 3},
    }
    resp = client.post(f"/api/v1/projects/{PROJECT_ID}/sources", json=payload)
    assert resp.status_code == expected_status
    if behavior == "success":
        data = resp.json()
        assert data["id"] == str(CREATED_ID)
        assert data["active"] is True
        assert data["config"]["source_type"] == "usb_camera"
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
            assert update_data.config.source_type == SourceType.USB_CAMERA
            if behavior == "success":
                return make_source_schema(source_id, update_data.config.device_id, update_data.active)
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
        "active": False,
        "config": {"source_type": "usb_camera", "device_id": 7},
    }
    resp = client.put(f"/api/v1/projects/{PROJECT_ID}/sources/{SOURCE_ID_1}", json=payload)
    assert resp.status_code == expected_status
    if behavior == "success":
        data = resp.json()
        assert data["id"] == str(SOURCE_ID_1)
        assert data["config"]["device_id"] == 7
        assert data["config"]["source_type"] == "usb_camera"
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


@pytest.mark.parametrize(
    "behavior,expected_status,expected_total",
    [
        ("success", 200, 100),
        ("not_active", 400, None),
        ("not_seekable", 400, None),
        ("no_pipeline", 400, None),
        ("source_not_found", 404, None),
        ("error", 500, None),
    ],
)
def test_get_frames(client, behavior, expected_status, expected_total):
    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def get_source(self, project_id: UUID, source_id: UUID):
            assert project_id == PROJECT_ID
            assert source_id == SOURCE_ID_1
            if behavior == "source_not_found":
                raise ResourceNotFoundError(ResourceType.SOURCE, str(source_id))
            active = behavior not in ("not_active",)
            return make_source_schema(SOURCE_ID_1, 0, active)

    class FakePipelineManager:
        def list_frames(self, project_id: UUID, offset: int = 0, limit: int = 30):
            assert project_id == PROJECT_ID
            if behavior == "success":
                frames = [
                    FrameMetadata(index=i, thumbnail=f"data:image/jpeg;base64,thumb_{i}")
                    for i in range(offset, min(offset + limit, 100))
                ]
                pagination = Pagination(count=len(frames), total=100, offset=offset, limit=limit)
                return FrameListResponse(frames=frames, pagination=pagination)
            if behavior == "not_seekable":
                raise SourceNotSeekableError("The active source does not support frame listing.")
            if behavior == "no_pipeline":
                raise PipelineNotActiveError("No active pipeline.")
            if behavior == "error":
                raise RuntimeError("Pipeline error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_source_service] = lambda: FakeService(None, None)
    client.app.dependency_overrides[get_pipeline_manager] = lambda: FakePipelineManager()

    resp = client.get(f"/api/v1/projects/{PROJECT_ID}/sources/{SOURCE_ID_1}/frames?offset=0&limit=30")
    assert resp.status_code == expected_status
    if behavior == "success":
        data = resp.json()
        assert data["pagination"]["total"] == expected_total
        assert data["pagination"]["offset"] == 0
        assert data["pagination"]["limit"] == 30
        assert data["pagination"]["count"] == 30
        assert len(data["frames"]) == 30
        assert data["frames"][0]["index"] == 0
        assert "thumbnail" in data["frames"][0]
        assert data["frames"][0]["thumbnail"].startswith("data:image/jpeg;base64,")
    else:
        assert "detail" in resp.json()


@pytest.mark.parametrize(
    "behavior,expected_status,expected_index",
    [
        ("success", 200, 42),
        ("not_active", 400, None),
        ("not_seekable", 400, None),
        ("no_pipeline", 400, None),
        ("source_not_found", 404, None),
        ("error", 500, None),
    ],
)
def test_get_frame_index(client, behavior, expected_status, expected_index):
    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def get_source(self, project_id: UUID, source_id: UUID):
            assert project_id == PROJECT_ID
            assert source_id == SOURCE_ID_1
            if behavior == "source_not_found":
                raise ResourceNotFoundError(ResourceType.SOURCE, str(source_id))
            active = behavior not in ("not_active",)
            return make_source_schema(SOURCE_ID_1, 0, active)

    class FakePipelineManager:
        def get_frame_index(self, project_id: UUID):
            assert project_id == PROJECT_ID
            if behavior == "success":
                return 42
            if behavior == "not_seekable":
                raise SourceNotSeekableError("The active source does not support frame indexing.")
            if behavior == "no_pipeline":
                raise PipelineNotActiveError("No active pipeline.")
            if behavior == "error":
                raise RuntimeError("Pipeline error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_source_service] = lambda: FakeService(None, None)
    client.app.dependency_overrides[get_pipeline_manager] = lambda: FakePipelineManager()

    resp = client.get(f"/api/v1/projects/{PROJECT_ID}/sources/{SOURCE_ID_1}/frames/index")
    assert resp.status_code == expected_status
    if behavior == "success":
        data = resp.json()
        assert data["index"] == expected_index
    else:
        assert "detail" in resp.json()


@pytest.mark.parametrize(
    "behavior,expected_status,seek_index",
    [
        ("success", 200, 10),
        ("not_active", 400, 10),
        ("not_seekable", 400, 10),
        ("out_of_bounds", 400, 999),
        ("no_pipeline", 400, 10),
        ("source_not_found", 404, 10),
        ("error", 500, 10),
    ],
)
def test_seek_frame(client, behavior, expected_status, seek_index):  # noqa: C901
    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def get_source(self, project_id: UUID, source_id: UUID):
            assert project_id == PROJECT_ID
            assert source_id == SOURCE_ID_1
            if behavior == "source_not_found":
                raise ResourceNotFoundError(ResourceType.SOURCE, str(source_id))
            active = behavior not in ("not_active",)
            return make_source_schema(SOURCE_ID_1, 0, active)

    class FakePipelineManager:
        def seek(self, project_id: UUID, index: int):
            assert project_id == PROJECT_ID
            assert index == seek_index
            if behavior == "success":
                return
            if behavior == "not_seekable":
                raise SourceNotSeekableError("The active source does not support frame navigation.")
            if behavior == "out_of_bounds":
                raise IndexError(f"Index {index} out of range")
            if behavior == "no_pipeline":
                raise PipelineNotActiveError("No active pipeline.")
            if behavior == "error":
                raise RuntimeError("Pipeline error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_source_service] = lambda: FakeService(None, None)
    client.app.dependency_overrides[get_pipeline_manager] = lambda: FakePipelineManager()

    resp = client.post(f"/api/v1/projects/{PROJECT_ID}/sources/{SOURCE_ID_1}/frames/{seek_index}")
    assert resp.status_code == expected_status
    if behavior == "success":
        data = resp.json()
        assert data["index"] == seek_index
    else:
        assert "detail" in resp.json()


def test_seek_frame_not_connected_error_message(client):
    """Test that seeking an inactive source returns appropriate error message."""

    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def get_source(self, project_id: UUID, source_id: UUID):
            return make_source_schema(SOURCE_ID_1, 0, active=False)

    class FakePipelineManager:
        def seek(self, project_id: UUID, index: int):
            pass

    client.app.dependency_overrides[get_source_service] = lambda: FakeService(None, None)
    client.app.dependency_overrides[get_pipeline_manager] = lambda: FakePipelineManager()

    resp = client.post(f"/api/v1/projects/{PROJECT_ID}/sources/{SOURCE_ID_1}/frames/10")
    assert resp.status_code == 400
    data = resp.json()
    assert "not currently active" in data["detail"]
    assert str(SOURCE_ID_1) in data["detail"]
