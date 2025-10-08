# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.components.schemas.reader import WebCamConfig
from dependencies import SessionDep  # type: ignore
from rest.endpoints import projects
from routers import projects_router
from services.common import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
)

PROJECT_ID = uuid4()
PROJECT_ID_STR = str(PROJECT_ID)
SECOND_PROJECT_ID = uuid4()
SECOND_PROJECT_ID_STR = str(SECOND_PROJECT_ID)
SOURCE_ID = uuid4()
PROCESSOR_ID = uuid4()
SINK_ID = uuid4()


@dataclass
class FakeSource:
    id: UUID
    config: object
    connected: bool = False


@dataclass
class FakeProcessor:
    id: UUID
    type: str
    config: dict
    name: str


@dataclass
class FakeSink:
    id: UUID
    config: dict


@dataclass
class FakeProject:
    id: UUID
    name: str
    sources: list[FakeSource] | None = None
    processor: FakeProcessor | None = None
    sink: FakeSink | None = None


def make_source() -> FakeSource:
    return FakeSource(
        id=SOURCE_ID,
        config=WebCamConfig(device_id=12345, source_type="webcam"),
        connected=False,
    )


def make_processor() -> FakeProcessor:
    return FakeProcessor(
        id=PROCESSOR_ID,
        type="DUMMY",
        config={"mode": "fast"},
        name="proc1",
    )


def make_sink() -> FakeSink:
    return FakeSink(id=SINK_ID, config={"dest": "stdout"})


def make_project(
    project_id: UUID,
    name: str,
    source: FakeSource | None = None,
    processor: FakeProcessor | None = None,
    sink: FakeSink | None = None,
) -> FakeProject:
    sources = [source] if source else []
    return FakeProject(
        id=project_id,
        name=name,
        sources=sources,
        processor=processor,
        sink=sink,
    )


def assert_minimal_project_payload(data: dict, project_id: str, name: str):
    assert data["id"] == project_id
    assert data["name"] == name
    assert data["sources"] == []
    assert data["processor"] is None
    assert data["sink"] is None


def assert_full_project_payload(data: dict):
    assert data["id"] == PROJECT_ID_STR
    assert data["name"] == "fullproj"
    assert isinstance(data["sources"], list) and len(data["sources"]) == 1
    src = data["sources"][0]
    assert src["id"] == str(SOURCE_ID)
    assert src["config"]["source_type"] == "webcam"
    assert src["config"]["device_id"] == 12345
    assert src["connected"] is False
    assert data["processor"] == {
        "id": str(PROCESSOR_ID),
        "type": "DUMMY",
        "config": {"mode": "fast"},
        "name": "proc1",
    }
    assert data["sink"] == {
        "id": str(SINK_ID),
        "config": {"dest": "stdout"},
    }


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(projects_router, prefix="/api/v1")
    app.dependency_overrides[SessionDep] = lambda: object()
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
        def __init__(self, session):
            pass

        def create_project(self, project_entity):
            assert project_entity.name == "myproj"
            if behavior == "success":
                return make_project(PROJECT_ID, "myproj")
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
        def __init__(self, session):
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
        def __init__(self, session):
            pass

        def get_active_project(self):
            if behavior == "success":
                return make_project(PROJECT_ID, "activeproj")
            if behavior == "notfound":
                raise ResourceNotFoundError(
                    resource_type=ResourceType.PROJECT,
                    resource_id="active",
                )
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(ep_mod, "ProjectService", FakeService)

    resp = client.get("/api/v1/projects/active")
    assert resp.status_code == expected_status
    if expected_detail:
        assert resp.json()["detail"] == expected_detail
    elif behavior == "success":
        assert_minimal_project_payload(resp.json(), PROJECT_ID_STR, "activeproj")


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
        def __init__(self, session):
            pass

        def list_projects(self):
            if behavior == "no_projects":
                return []
            if behavior == "some_projects":
                return [
                    make_project(PROJECT_ID, "proj1"),
                    make_project(SECOND_PROJECT_ID, "proj2"),
                ]
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
    projects = data["projects"]
    assert len(projects) == expected_count
    if behavior == "some_projects":
        ids = {p["id"] for p in projects}
        assert ids == {PROJECT_ID_STR, SECOND_PROJECT_ID_STR}
        names = {p["id"]: p["name"] for p in projects}
        assert names[PROJECT_ID_STR] == "proj1"
        assert names[SECOND_PROJECT_ID_STR] == "proj2"


@pytest.mark.parametrize(
    "behavior,expected_status,expect_payload_type",
    [
        ("minimal", 200, "minimal"),
        ("full", 200, "full"),
        ("notfound", 404, None),
        ("error", 500, None),
    ],
)
def test_get_project(client, monkeypatch, behavior, expected_status, expect_payload_type):  # noqa: C901
    from rest.endpoints import projects as ep_mod

    class FakeService:
        def __init__(self, session):
            pass

        def get_project(self, project_id: UUID):
            assert project_id == PROJECT_ID
            if behavior == "minimal":
                return make_project(PROJECT_ID, "minproj")
            if behavior == "full":
                return make_project(
                    PROJECT_ID,
                    "fullproj",
                    source=make_source(),
                    processor=make_processor(),
                    sink=make_sink(),
                )
            if behavior == "notfound":
                raise ResourceNotFoundError(
                    resource_type=ResourceType.PROJECT,
                    resource_id=str(project_id),
                )
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(ep_mod, "ProjectService", FakeService)
    resp = client.get(f"/api/v1/projects/{PROJECT_ID_STR}")

    assert resp.status_code == expected_status

    if expect_payload_type is None:
        if behavior == "notfound":
            assert resp.status_code == 404
        elif behavior == "error":
            assert resp.json()["detail"] == "Failed to retrieve project."
        return

    data = resp.json()
    if expect_payload_type == "minimal":
        assert_minimal_project_payload(data, PROJECT_ID_STR, "minproj")
    if expect_payload_type == "full":
        assert_full_project_payload(data)


@pytest.mark.parametrize(
    "behavior,expected_status,expect_detail",
    [
        ("success", 200, None),
        ("notfound", 404, "notfound"),  # marker; we only check status
        ("error", 500, "Failed to update project due to internal server error."),
    ],
)
def test_update_project(client, monkeypatch, behavior, expected_status, expect_detail):
    from rest.endpoints import projects as ep_mod

    NEW_NAME = "renamed"

    class FakeService:
        def __init__(self, session):
            pass

        def update_project(self, project_id: UUID, new_name: str):
            assert project_id == PROJECT_ID
            assert new_name == NEW_NAME
            if behavior == "success":
                return make_project(PROJECT_ID, NEW_NAME)
            if behavior == "notfound":
                raise ResourceNotFoundError(
                    resource_type=ResourceType.PROJECT,
                    resource_id=str(project_id),
                )
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(ep_mod, "ProjectService", FakeService)

    resp = client.put(f"/api/v1/projects/{PROJECT_ID_STR}", json={"name": NEW_NAME})
    assert resp.status_code == expected_status
    if behavior == "success":
        assert_minimal_project_payload(resp.json(), PROJECT_ID_STR, NEW_NAME)
    elif behavior == "error":
        assert resp.json()["detail"] == expect_detail
    elif behavior == "notfound":
        # Do not assert exact message; just ensure 404 and detail present.
        assert "detail" in resp.json()


def test_update_project_validation_error(client):
    resp = client.put(f"/api/v1/projects/{PROJECT_ID_STR}", json={"name": ""})
    assert resp.status_code == 422
