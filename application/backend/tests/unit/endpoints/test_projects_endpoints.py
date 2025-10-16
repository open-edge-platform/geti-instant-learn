# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dependencies import SessionDep, get_config_dispatcher  # type: ignore
from rest.endpoints import projects
from routers import projects_router
from services.errors import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
)
from services.schemas.project import ProjectSchema, ProjectsListSchema

PROJECT_ID = uuid4()
PROJECT_ID_STR = str(PROJECT_ID)
SECOND_PROJECT_ID = uuid4()
SECOND_PROJECT_ID_STR = str(SECOND_PROJECT_ID)


def assert_project_schema(data: dict, project_id: str, name: str):
    assert data["id"] == project_id
    assert data["name"] == name


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(projects_router, prefix="/api/v1")
    app.dependency_overrides[SessionDep] = lambda: object()

    class DummyDispatcher:
        def dispatch(self, event):  # noqa: D401
            pass

    app.dependency_overrides[get_config_dispatcher] = lambda: DummyDispatcher()
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.mark.parametrize(
    "behavior,expected_status,expect_location,expect_conflict_substring",
    [
        ("success", 201, True, None),
        ("conflict", 409, False, PROJECT_ID_STR),
        ("error", 500, False, None),
    ],
)
def test_create_project(client, monkeypatch, behavior, expected_status, expect_location, expect_conflict_substring):
    class FakeService:
        def __init__(self, session, config_change_dispatcher):  # noqa: D401
            pass

        def create_project(self, payload):
            assert payload.name == "myproj"
            if behavior == "success":
                return ProjectSchema(id=PROJECT_ID, name="myproj")
            if behavior == "conflict":
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.PROJECT,
                    resource_value=str(PROJECT_ID),
                    raised_by="id",
                )
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(projects, "ProjectService", FakeService)

    payload = {"id": PROJECT_ID_STR, "name": "myproj"}
    resp = client.post("/api/v1/projects", json=payload)

    assert resp.status_code == expected_status
    if expect_location:
        assert resp.headers.get("Location") == f"/projects/{PROJECT_ID_STR}"
        assert resp.text == ""
    else:
        assert "Location" not in resp.headers
        if behavior == "conflict":
            assert expect_conflict_substring in resp.json()["detail"]
        if behavior == "error":
            assert resp.json()["detail"] == "Failed to create a project due to internal server error."


@pytest.mark.parametrize(
    "behavior,expected_status,expect_detail",
    [
        ("success", 204, None),
        ("missing", 204, None),  # Endpoint suppresses 404
        ("error", 500, f"Failed to delete project with id {PROJECT_ID_STR} due to an internal error."),
    ],
)
def test_delete_project(client, monkeypatch, behavior, expected_status, expect_detail):
    from rest.endpoints import projects as ep_mod

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
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(ep_mod, "ProjectService", FakeService)

    resp = client.delete(f"/api/v1/projects/{PROJECT_ID_STR}")
    assert resp.status_code == expected_status
    if expect_detail:
        assert resp.json()["detail"] == expect_detail
    else:
        assert resp.text == ""


@pytest.mark.parametrize(
    "behavior,expected_status,expected_detail",
    [
        ("success", 200, None),
        ("notfound", 404, "No active project found."),
        ("error", 500, "Failed to retrieve active project."),
    ],
)
def test_get_active_project(client, monkeypatch, behavior, expected_status, expected_detail):
    from rest.endpoints import projects as ep_mod

    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def get_active_project_info(self):
            if behavior == "success":
                return ProjectSchema(id=PROJECT_ID, name="activeproj")
            if behavior == "notfound":
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id="active")
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(ep_mod, "ProjectService", FakeService)
    resp = client.get("/api/v1/projects/active")
    assert resp.status_code == expected_status
    if expected_detail:
        assert resp.json()["detail"] == expected_detail
    elif behavior == "success":
        assert_project_schema(resp.json(), PROJECT_ID_STR, "activeproj")


@pytest.mark.parametrize(
    "behavior,expected_status,expected_count,expected_detail",
    [
        ("no_projects", 200, 0, None),
        ("some_projects", 200, 2, None),
        ("error", 500, None, "Failed to list projects."),
    ],
)
def test_get_projects_list(client, monkeypatch, behavior, expected_status, expected_count, expected_detail):
    from rest.endpoints import projects as ep_mod

    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def list_projects(self):
            if behavior == "no_projects":
                return ProjectsListSchema(projects=[])
            if behavior == "some_projects":
                return ProjectsListSchema(
                    projects=[
                        ProjectSchema(id=PROJECT_ID, name="proj1"),
                        ProjectSchema(id=SECOND_PROJECT_ID, name="proj2"),
                    ]
                )
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(ep_mod, "ProjectService", FakeService)
    resp = client.get("/api/v1/projects")
    assert resp.status_code == expected_status
    if expected_detail:
        assert resp.json()["detail"] == expected_detail
        return
    data = resp.json()
    projects_list = data["projects"]
    assert len(projects_list) == expected_count
    if behavior == "some_projects":
        ids = {p["id"] for p in projects_list}
        assert ids == {PROJECT_ID_STR, SECOND_PROJECT_ID_STR}


@pytest.mark.parametrize(
    "behavior,expected_status,expect_payload",
    [
        ("minimal", 200, True),
        ("notfound", 404, False),
        ("error", 500, False),
    ],
)
def test_get_project(client, monkeypatch, behavior, expected_status, expect_payload):
    from rest.endpoints import projects as ep_mod

    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def get_project(self, project_id: UUID):
            assert project_id == PROJECT_ID
            if behavior == "minimal":
                return ProjectSchema(id=PROJECT_ID, name="minproj")
            if behavior == "notfound":
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(ep_mod, "ProjectService", FakeService)
    resp = client.get(f"/api/v1/projects/{PROJECT_ID_STR}")
    assert resp.status_code == expected_status
    if expect_payload:
        assert_project_schema(resp.json(), PROJECT_ID_STR, "minproj")
    else:
        if behavior == "error":
            assert resp.json()["detail"] == "Failed to retrieve project."


@pytest.mark.parametrize(
    "behavior,expected_status,expect_detail",
    [
        ("success", 200, None),
        ("notfound", 404, "nf"),
        ("error", 500, "Failed to update project due to internal server error."),
    ],
)
def test_update_project(client, monkeypatch, behavior, expected_status, expect_detail):
    from rest.endpoints import projects as ep_mod

    NEW_NAME = "renamed"

    class FakeService:
        def __init__(self, session, config_change_dispatcher):
            pass

        def update_project(self, project_id: UUID, update_data):
            assert project_id == PROJECT_ID
            assert update_data.name == NEW_NAME
            if behavior == "success":
                return ProjectSchema(id=PROJECT_ID, name=NEW_NAME)
            if behavior == "notfound":
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(ep_mod, "ProjectService", FakeService)
    resp = client.put(f"/api/v1/projects/{PROJECT_ID_STR}", json={"name": NEW_NAME})
    assert resp.status_code == expected_status
    if behavior == "success":
        assert_project_schema(resp.json(), PROJECT_ID_STR, NEW_NAME)
    elif behavior == "error":
        assert resp.json()["detail"] == expect_detail
    elif behavior == "notfound":
        assert "detail" in resp.json()


def test_update_project_validation_error(client):
    resp = client.put(f"/api/v1/projects/{PROJECT_ID_STR}", json={"name": ""})
    assert resp.status_code == 422
