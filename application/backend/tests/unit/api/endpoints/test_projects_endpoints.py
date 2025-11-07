# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from api.error_handler import custom_exception_handler
from api.routers import projects_router
from dependencies import SessionDep, get_config_dispatcher, get_project_service
from domain.errors import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
)
from domain.services.schemas.base import Pagination
from domain.services.schemas.project import ProjectSchema, ProjectsListSchema

PROJECT_ID = uuid4()
PROJECT_ID_STR = str(PROJECT_ID)
SECOND_PROJECT_ID = uuid4()
SECOND_PROJECT_ID_STR = str(SECOND_PROJECT_ID)


def assert_project_schema(data: dict, project_id: str, name: str, active: bool = False):
    assert data["id"] == project_id
    assert data["name"] == name
    assert data["active"] == active


@pytest.fixture
def app():
    from api.endpoints import projects as _  # noqa: F401

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
    "behavior,expected_status,expect_location,expect_substring",
    [
        ("success", 201, True, None),
        ("conflict", 409, False, "already exists"),
        ("error", 500, False, "internal server error"),
    ],
)
def test_create_project(client, behavior, expected_status, expect_location, expect_substring):
    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def create_project(self, payload):
            assert payload.name == "myproj"
            if behavior == "success":
                return ProjectSchema(id=PROJECT_ID, name="myproj", active=True)
            if behavior == "conflict":
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.PROJECT,
                    resource_value="myproj",
                    field="name",
                    message="A project with this name already exists.",
                )
            if behavior == "error":
                raise RuntimeError("Database connection failed")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_project_service] = lambda: FakeService(None, None)

    payload = {"id": PROJECT_ID_STR, "name": "myproj"}
    resp = client.post("/api/v1/projects", json=payload)

    assert resp.status_code == expected_status
    if expect_location:
        assert resp.headers.get("Location") == f"/projects/{PROJECT_ID_STR}"
        response_data = resp.json()
        assert_project_schema(response_data, PROJECT_ID_STR, "myproj", active=True)
    else:
        assert "Location" not in resp.headers
        if expect_substring:
            assert expect_substring.lower() in resp.json()["detail"].lower()


@pytest.mark.parametrize(
    "behavior,expected_status",
    [
        ("success", 204),
        ("missing", 404),
        ("error", 500),
    ],
)
def test_delete_project(client, behavior, expected_status):
    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def delete_project(self, project_id: UUID):
            assert project_id == PROJECT_ID
            if behavior == "success":
                return
            if behavior == "missing":
                raise ResourceNotFoundError(
                    resource_type=ResourceType.PROJECT,
                    resource_id=str(project_id),
                )
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_project_service] = lambda: FakeService(None, None)

    resp = client.delete(f"/api/v1/projects/{PROJECT_ID_STR}")
    assert resp.status_code == expected_status
    if expected_status == 204:
        assert resp.text == ""
    else:
        assert "detail" in resp.json()


@pytest.mark.parametrize(
    "behavior,expected_status",
    [
        ("success", 200),
        ("notfound", 404),
        ("error", 500),
    ],
)
def test_get_active_project(client, behavior, expected_status):
    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def get_active_project_info(self):
            if behavior == "success":
                return ProjectSchema(id=PROJECT_ID, name="activeproj", active=True)
            if behavior == "notfound":
                raise ResourceNotFoundError(
                    resource_type=ResourceType.PROJECT,
                    resource_id=None,
                    message="No active project found.",
                )
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_project_service] = lambda: FakeService(None, None)

    resp = client.get("/api/v1/projects/active")
    assert resp.status_code == expected_status
    if behavior == "success":
        assert_project_schema(resp.json(), PROJECT_ID_STR, "activeproj", active=True)
    else:
        assert "detail" in resp.json()


@pytest.mark.parametrize(
    "behavior,expected_status,expected_count",
    [
        ("no_projects", 200, 0),
        ("some_projects", 200, 2),
        ("error", 500, None),
    ],
)
def test_get_projects_list(client, behavior, expected_status, expected_count):
    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def list_projects(self, offset=0, limit=20):
            if behavior == "no_projects":
                return ProjectsListSchema(
                    projects=[], pagination=Pagination(count=0, total=0, offset=offset, limit=limit)
                )
            if behavior == "some_projects":
                projects = [
                    ProjectSchema(id=PROJECT_ID, name="proj1", active=False),
                    ProjectSchema(id=SECOND_PROJECT_ID, name="proj2", active=False),
                ]
                return ProjectsListSchema(
                    projects=projects, pagination=Pagination(count=2, total=2, offset=offset, limit=limit)
                )
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_project_service] = lambda: FakeService(None, None)

    resp = client.get("/api/v1/projects")
    assert resp.status_code == expected_status
    if behavior == "error":
        assert "detail" in resp.json()
        return

    data = resp.json()
    projects_list = data["projects"]
    assert len(projects_list) == expected_count
    assert "pagination" in data
    pagination = data["pagination"]
    assert pagination["count"] == expected_count
    assert pagination["offset"] == 0
    assert pagination["limit"] == 20
    if behavior == "some_projects":
        ids = {p["id"] for p in projects_list}
        assert ids == {PROJECT_ID_STR, SECOND_PROJECT_ID_STR}


def test_get_projects_list_with_pagination_params(client):
    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def list_projects(self, offset=0, limit=20):
            assert offset == 10
            assert limit == 5
            projects = [ProjectSchema(id=PROJECT_ID, name="proj1", active=False)]
            return ProjectsListSchema(projects=projects, pagination=Pagination(count=1, total=15, offset=10, limit=5))

    client.app.dependency_overrides[get_project_service] = lambda: FakeService(None, None)

    resp = client.get("/api/v1/projects?offset=10&limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["projects"]) == 1
    pagination = data["pagination"]
    assert pagination["count"] == 1
    assert pagination["total"] == 15
    assert pagination["offset"] == 10
    assert pagination["limit"] == 5


@pytest.mark.parametrize(
    "behavior,expected_status,expect_payload",
    [
        ("minimal", 200, True),
        ("notfound", 404, False),
        ("error", 500, False),
    ],
)
def test_get_project(client, behavior, expected_status, expect_payload):
    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def get_project(self, project_id: UUID):
            assert project_id == PROJECT_ID
            if behavior == "minimal":
                return ProjectSchema(id=PROJECT_ID, name="minproj", active=False)
            if behavior == "notfound":
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_project_service] = lambda: FakeService(None, None)

    resp = client.get(f"/api/v1/projects/{PROJECT_ID_STR}")
    assert resp.status_code == expected_status
    if expect_payload:
        assert_project_schema(resp.json(), PROJECT_ID_STR, "minproj", active=False)
    else:
        assert "detail" in resp.json()


@pytest.mark.parametrize(
    "behavior,expected_status",
    [
        ("success", 200),
        ("notfound", 404),
        ("conflict", 409),
        ("error", 500),
    ],
)
def test_update_project(client, behavior, expected_status):
    NEW_NAME = "renamed"

    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def update_project(self, project_id: UUID, update_data):
            assert project_id == PROJECT_ID
            assert update_data.name == NEW_NAME
            if behavior == "success":
                return ProjectSchema(id=PROJECT_ID, name=NEW_NAME, active=False)
            if behavior == "notfound":
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))
            if behavior == "conflict":
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.PROJECT,
                    resource_value=NEW_NAME,
                    field="name",
                    message="A project with this name already exists.",
                )
            if behavior == "error":
                raise RuntimeError("Database error")
            raise AssertionError("Unhandled behavior")

    client.app.dependency_overrides[get_project_service] = lambda: FakeService(None, None)

    resp = client.put(f"/api/v1/projects/{PROJECT_ID_STR}", json={"name": NEW_NAME, "active": False})
    assert resp.status_code == expected_status
    if behavior == "success":
        assert_project_schema(resp.json(), PROJECT_ID_STR, NEW_NAME, active=False)
    else:
        assert "detail" in resp.json()


def test_update_project_validation_error(client):
    resp = client.put(f"/api/v1/projects/{PROJECT_ID_STR}", json={"name": ""})
    assert resp.status_code == 400
    assert "detail" in resp.json()
