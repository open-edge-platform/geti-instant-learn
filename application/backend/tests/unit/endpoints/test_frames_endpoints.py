# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dependencies import (
    SessionDep,
    get_frame_repository,
    get_frame_service,
    get_pipeline_manager,
    get_project_repository,
    get_source_repository,
)
from routers import projects_router
from services.errors import ResourceNotFoundError, ResourceType, ServiceError

PROJECT_ID = uuid4()
PROJECT_ID_STR = str(PROJECT_ID)
FRAME_ID = uuid4()
FRAME_ID_STR = str(FRAME_ID)
SECOND_FRAME_ID = uuid4()
SECOND_FRAME_ID_STR = str(SECOND_FRAME_ID)


@pytest.fixture
def app():
    from rest.endpoints import frames as _  # noqa: F401

    app = FastAPI()
    app.include_router(projects_router, prefix="/api/v1")

    # Override nested dependencies to prevent AttributeError
    app.dependency_overrides[SessionDep] = lambda: object()
    app.dependency_overrides[get_pipeline_manager] = lambda: object()
    app.dependency_overrides[get_frame_repository] = lambda: object()
    app.dependency_overrides[get_project_repository] = lambda: object()
    app.dependency_overrides[get_source_repository] = lambda: object()

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def _get_capture_frame_exception(behavior, project_id):
    if behavior == "project_not_found":
        return ResourceNotFoundError(
            resource_type=ResourceType.PROJECT,
            resource_id=str(project_id),
        )
    if behavior == "source_not_found":
        return ResourceNotFoundError(
            resource_type=ResourceType.SOURCE,
            resource_id=f"no connected source for project {project_id}",
        )
    if behavior == "project_not_active":
        return ServiceError(f"project {project_id} is not active")
    if behavior == "capture_timeout":
        return ServiceError("No frame received within 5.0 seconds")
    if behavior == "unexpected_error":
        return RuntimeError("Database connection failed")
    return None


@pytest.mark.parametrize(
    "behavior,expected_status,expect_location,expected_detail",
    [
        ("success", 201, True, None),
        ("project_not_found", 404, False, None),
        ("source_not_found", 404, False, None),
        ("project_not_active", 400, False, None),
        ("capture_timeout", 400, False, None),
        ("unexpected_error", 500, False, "An unexpected error occurred while capturing the frame"),
    ],
)
def test_capture_frame(client, monkeypatch, behavior, expected_status, expect_location, expected_detail):
    class FakeFrameService:
        def __init__(self, pipeline_manager, frame_repo, project_repo, source_repo):
            pass

        def capture_frame(self, project_id):
            assert project_id == PROJECT_ID
            if behavior == "success":
                return FRAME_ID

            exception = _get_capture_frame_exception(behavior, project_id)
            if exception:
                raise exception
            raise AssertionError("Unhandled behavior")

    app = client.app
    app.dependency_overrides[get_frame_service] = lambda: FakeFrameService(None, None, None, None)

    resp = client.post(f"/api/v1/projects/{PROJECT_ID_STR}/frames")

    assert resp.status_code == expected_status

    if expect_location:
        assert resp.headers.get("Location") == f"/projects/{PROJECT_ID_STR}/frames/{FRAME_ID_STR}"
        assert resp.json()["frame_id"] == FRAME_ID_STR
    else:
        assert "Location" not in resp.headers
        if expected_detail:
            assert expected_detail in resp.json()["detail"]
        else:
            assert "detail" in resp.json()


def test_capture_frame_service_error_details(client, monkeypatch):
    class FakeFrameService:
        def __init__(self, pipeline_manager, frame_repo, project_repo, source_repo):
            pass

        def capture_frame(self, project_id):
            raise ServiceError("Frame capture failed: encoding error")

    app = client.app
    app.dependency_overrides[get_frame_service] = lambda: FakeFrameService(None, None, None, None)

    resp = client.post(f"/api/v1/projects/{PROJECT_ID_STR}/frames")

    assert resp.status_code == 400
    assert "Frame capture failed: encoding error" in resp.json()["detail"]


@pytest.mark.parametrize(
    "frame_exists,path_exists,expected_status",
    [
        (True, True, 200),
        (True, False, 404),
        (False, False, 404),
    ],
)
def test_get_frame(client, monkeypatch, frame_exists, path_exists, expected_status, tmp_path):
    test_frame_path = tmp_path / "test_frame.jpg"
    if path_exists:
        test_frame_path.write_bytes(b"\xff\xd8\xff\xe0")  # minimal JPEG header

    class FakeFrameService:
        def __init__(self, pipeline_manager, frame_repo, project_repo, source_repo):
            pass

        def get_frame_path(self, project_id, frame_id):
            assert project_id == PROJECT_ID
            assert frame_id == FRAME_ID
            if frame_exists:
                return test_frame_path
            return None

    app = client.app
    app.dependency_overrides[get_frame_service] = lambda: FakeFrameService(None, None, None, None)

    resp = client.get(f"/api/v1/projects/{PROJECT_ID_STR}/frames/{FRAME_ID_STR}")

    assert resp.status_code == expected_status

    if expected_status == 200:
        assert resp.headers["content-type"] == "image/jpeg"
        assert len(resp.content) > 0
    else:
        assert resp.json()["detail"] == "Frame not found"


def test_get_frame_returns_file_content(client, monkeypatch, tmp_path):
    test_frame_path = tmp_path / "frame.jpg"
    expected_content = b"\xff\xd8\xff\xe0\x00\x10JFIF"  # JPEG header with JFIF marker
    test_frame_path.write_bytes(expected_content)

    class FakeFrameService:
        def __init__(self, pipeline_manager, frame_repo, project_repo, source_repo):
            pass

        def get_frame_path(self, project_id, frame_id):
            return test_frame_path

    app = client.app
    app.dependency_overrides[get_frame_service] = lambda: FakeFrameService(None, None, None, None)

    resp = client.get(f"/api/v1/projects/{PROJECT_ID_STR}/frames/{FRAME_ID_STR}")

    assert resp.status_code == 200
    assert resp.content == expected_content
    assert resp.headers["content-type"] == "image/jpeg"


def test_get_frame_with_invalid_project_id(client, monkeypatch):
    resp = client.get(f"/api/v1/projects/not-a-uuid/frames/{FRAME_ID_STR}")
    assert resp.status_code == 422


def test_get_frame_with_invalid_frame_id(client, monkeypatch):
    resp = client.get(f"/api/v1/projects/{PROJECT_ID_STR}/frames/not-a-uuid")
    assert resp.status_code == 422


def test_capture_frame_with_invalid_project_id(client):
    resp = client.post("/api/v1/projects/not-a-uuid/frames")
    assert resp.status_code == 422


def test_capture_multiple_frames_returns_different_ids(client, monkeypatch):
    frame_ids = [FRAME_ID, SECOND_FRAME_ID]
    call_count = 0

    class FakeFrameService:
        def __init__(self, pipeline_manager, frame_repo, project_repo, source_repo):
            pass

        def capture_frame(self, project_id):
            nonlocal call_count
            result = frame_ids[call_count]
            call_count += 1
            return result

    app = client.app
    app.dependency_overrides[get_frame_service] = lambda: FakeFrameService(None, None, None, None)

    resp1 = client.post(f"/api/v1/projects/{PROJECT_ID_STR}/frames")
    assert resp1.status_code == 201
    assert resp1.headers["Location"] == f"/projects/{PROJECT_ID_STR}/frames/{FRAME_ID_STR}"
    assert resp1.json()["frame_id"] == FRAME_ID_STR

    resp2 = client.post(f"/api/v1/projects/{PROJECT_ID_STR}/frames")
    assert resp2.status_code == 201
    assert resp2.headers["Location"] == f"/projects/{PROJECT_ID_STR}/frames/{SECOND_FRAME_ID_STR}"
    assert resp2.json()["frame_id"] == SECOND_FRAME_ID_STR

    assert resp1.headers["Location"] != resp2.headers["Location"]
    assert resp1.json()["frame_id"] != resp2.json()["frame_id"]
