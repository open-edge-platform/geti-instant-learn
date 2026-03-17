# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from api.error_handler import custom_exception_handler
from api.routers import projects_router
from dependencies import SessionDep, get_config_dispatcher, get_license_service, get_project_service
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


def assert_project_schema(
    data: dict,
    project_id: str,
    name: str,
    active: bool = False,
    device: str = "cpu",
    prompt_mode: str = "visual",
):
    assert data["id"] == project_id
    assert data["name"] == name
    assert data["active"] == active
    assert data["config"]["device"] == device
    assert data["config"]["prompt_mode"] == prompt_mode


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


@pytest.fixture
def mock_license_service():
    return MagicMock()


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
                return ProjectSchema(id=PROJECT_ID, name="myproj", active=True, config={"device": "cpu"})
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


class TestCreateProject:
    """Tests for POST /projects endpoint."""

    @pytest.fixture(autouse=True)
    def override_license_service(self, app, mock_license_service):
        app.dependency_overrides[get_license_service] = lambda: mock_license_service
        yield
        app.dependency_overrides.pop(get_license_service, None)

    @pytest.mark.parametrize(
        "behavior,expected_status,expect_location,expect_substring",
        [
            ("success", 201, True, None),
            ("conflict", 409, False, "already exists"),
            ("error", 500, False, "internal server error"),
        ],
    )
    def test_create_project_with_license_accepted(
        self, app, client, mock_license_service, behavior, expected_status, expect_location, expect_substring
    ):
        """Test project creation with license accepted."""
        mock_license_service.is_accepted.return_value = True

        class FakeService:
            def __init__(self, session, config_change_dispatcher):
                pass

            def create_project(self, payload):
                assert payload.name == "myproj"
                if behavior == "success":
                    return ProjectSchema(id=PROJECT_ID, name="myproj", active=True, config={"device": "cpu"})
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

        app.dependency_overrides[get_project_service] = lambda: FakeService(None, None)

        payload = {"id": PROJECT_ID_STR, "name": "myproj"}
        resp = client.post("/api/v1/projects", json=payload)

        assert resp.status_code == expected_status
        mock_license_service.is_accepted.assert_called_once()
        if expect_location:
            assert resp.headers.get("Location") == f"/projects/{PROJECT_ID_STR}"
            response_data = resp.json()
            assert_project_schema(response_data, PROJECT_ID_STR, "myproj", active=True)
        else:
            assert "Location" not in resp.headers
            if expect_substring:
                assert expect_substring.lower() in resp.json()["detail"].lower()

    def test_create_project_license_not_accepted(self, app, client, mock_license_service):
        """Creating a project without accepting license returns 403."""
        mock_license_service.is_accepted.return_value = False

        class FakeService:
            def __init__(self, session, config_change_dispatcher):
                pass

            def create_project(self, payload):
                raise AssertionError("create_project should not be called when license is not accepted")

        app.dependency_overrides[get_project_service] = lambda: FakeService(None, None)

        payload = {"name": "myproj"}
        resp = client.post("/api/v1/projects", json=payload)

        assert resp.status_code == 403
        assert "license" in resp.json()["detail"].lower()
        mock_license_service.is_accepted.assert_called_once()


@pytest.mark.parametrize(
    "behavior,expected_status",
    [
        ("success", 204),
        ("missing", 404),
        ("error", 500),
    ],
)
def test_delete_project(app, client, behavior, expected_status):
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

    app.dependency_overrides[get_project_service] = lambda: FakeService(None, None)

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
                return ProjectSchema(id=PROJECT_ID, name="activeproj", active=True, config={"device": "cpu"})
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
                    ProjectSchema(id=PROJECT_ID, name="proj1", active=False, config={"device": "cpu"}),
                    ProjectSchema(id=SECOND_PROJECT_ID, name="proj2", active=False, config={"device": "cpu"}),
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
            projects = [ProjectSchema(id=PROJECT_ID, name="proj1", active=False, config={"device": "cpu"})]
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
                return ProjectSchema(id=PROJECT_ID, name="minproj", active=False, config={"device": "cpu"})
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
            assert update_data.config is not None
            assert update_data.config.device == "cuda"
            if behavior == "success":
                return ProjectSchema(id=PROJECT_ID, name=NEW_NAME, active=False, config={"device": "cuda"})
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

    resp = client.put(
        f"/api/v1/projects/{PROJECT_ID_STR}",
        json={"name": NEW_NAME, "active": False, "config": {"device": "cuda"}},
    )
    assert resp.status_code == expected_status
    if behavior == "success":
        assert_project_schema(resp.json(), PROJECT_ID_STR, NEW_NAME, active=False, device="cuda")
    else:
        assert "detail" in resp.json()


def test_update_project_validation_error(client):
    resp = client.put(f"/api/v1/projects/{PROJECT_ID_STR}", json={"name": ""})
    assert resp.status_code == 400
    assert "detail" in resp.json()


class TestPromptModeEndpoints:
    """Tests for prompt_mode field on GET and PUT /projects endpoints."""

    def test_get_project_returns_prompt_mode_text(self, client, app):
        """GET /{id} serializes prompt_mode=text in the response."""

        class FakeService:
            def __init__(self, session, config_change_dispatcher):
                pass

            def get_project(self, project_id: UUID):
                return ProjectSchema(
                    id=PROJECT_ID,
                    name="text_proj",
                    active=False,
                    config={"device": "cpu", "prompt_mode": "text"},
                )

        app.dependency_overrides[get_project_service] = lambda: FakeService(None, None)

        resp = client.get(f"/api/v1/projects/{PROJECT_ID_STR}")

        assert resp.status_code == 200
        assert_project_schema(resp.json(), PROJECT_ID_STR, "text_proj", prompt_mode="text")

    def test_get_project_returns_prompt_mode_visual(self, client, app):
        """GET /{id} serializes prompt_mode=visual in the response."""

        class FakeService:
            def __init__(self, session, config_change_dispatcher):
                pass

            def get_project(self, project_id: UUID):
                return ProjectSchema(
                    id=PROJECT_ID,
                    name="visual_proj",
                    active=False,
                    config={"device": "cpu", "prompt_mode": "visual"},
                )

        app.dependency_overrides[get_project_service] = lambda: FakeService(None, None)

        resp = client.get(f"/api/v1/projects/{PROJECT_ID_STR}")

        assert resp.status_code == 200
        assert_project_schema(resp.json(), PROJECT_ID_STR, "visual_proj", prompt_mode="visual")

    @pytest.mark.parametrize("prompt_mode", ["text", "visual"])
    def test_put_project_accepts_prompt_mode(self, client, app, prompt_mode):
        """PUT /{id} forwards prompt_mode to the service and reflects it in the response."""

        class FakeService:
            def __init__(self, session, config_change_dispatcher):
                pass

            def update_project(self, project_id: UUID, update_data):
                assert update_data.config is not None
                assert update_data.config.prompt_mode == prompt_mode
                return ProjectSchema(
                    id=project_id,
                    name="proj",
                    active=False,
                    config={"device": "cpu", "prompt_mode": prompt_mode},
                )

        app.dependency_overrides[get_project_service] = lambda: FakeService(None, None)

        resp = client.put(
            f"/api/v1/projects/{PROJECT_ID_STR}",
            json={"config": {"device": "cpu", "prompt_mode": prompt_mode}},
        )

        assert resp.status_code == 200
        assert resp.json()["config"]["prompt_mode"] == prompt_mode

    def test_put_project_invalid_prompt_mode_returns_400(self, client, app):
        """PUT /{id} with an unknown prompt_mode value returns 400."""
        app.dependency_overrides[get_project_service] = lambda: MagicMock()

        resp = client.put(
            f"/api/v1/projects/{PROJECT_ID_STR}",
            json={"config": {"prompt_mode": "invalid_mode"}},
        )

        assert resp.status_code == 400

    def test_put_project_prompt_mode_without_device_is_accepted(self, client, app):
        """PUT with only prompt_mode in config (no device) is valid."""

        class FakeService:
            def __init__(self, session, config_change_dispatcher):
                pass

            def update_project(self, project_id: UUID, update_data):
                assert update_data.config.prompt_mode == "text"
                return ProjectSchema(
                    id=project_id,
                    name="proj",
                    active=False,
                    config={"device": "cpu", "prompt_mode": "text"},
                )

        app.dependency_overrides[get_project_service] = lambda: FakeService(None, None)

        resp = client.put(
            f"/api/v1/projects/{PROJECT_ID_STR}",
            json={"config": {"prompt_mode": "text"}},
        )

        assert resp.status_code == 200
        assert resp.json()["config"]["prompt_mode"] == "text"

    def test_get_active_project_returns_prompt_mode(self, client, app):
        """GET /active includes prompt_mode in config."""

        class FakeService:
            def __init__(self, session, config_change_dispatcher):
                pass

            def get_active_project_info(self):
                return ProjectSchema(
                    id=PROJECT_ID,
                    name="active_proj",
                    active=True,
                    config={"device": "cpu", "prompt_mode": "visual"},
                )

        app.dependency_overrides[get_project_service] = lambda: FakeService(None, None)

        resp = client.get("/api/v1/projects/active")

        assert resp.status_code == 200
        assert resp.json()["config"]["prompt_mode"] == "visual"

    def test_list_projects_returns_prompt_mode_per_project(self, client, app):
        """GET / includes prompt_mode for each project in the list."""

        class FakeService:
            def __init__(self, session, config_change_dispatcher):
                pass

            def list_projects(self, offset=0, limit=20):
                return ProjectsListSchema(
                    projects=[
                        ProjectSchema(
                            id=PROJECT_ID,
                            name="p1",
                            active=False,
                            config={"device": "cpu", "prompt_mode": "text"},
                        ),
                        ProjectSchema(
                            id=SECOND_PROJECT_ID,
                            name="p2",
                            active=False,
                            config={"device": "cuda", "prompt_mode": "visual"},
                        ),
                    ],
                    pagination=Pagination(count=2, total=2, offset=0, limit=20),
                )

        app.dependency_overrides[get_project_service] = lambda: FakeService(None, None)

        resp = client.get("/api/v1/projects")

        assert resp.status_code == 200
        projects = {p["name"]: p["config"]["prompt_mode"] for p in resp.json()["projects"]}
        assert projects["p1"] == "text"
        assert projects["p2"] == "visual"
