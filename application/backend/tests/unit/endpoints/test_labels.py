# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest
from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic_extra_types.color import Color

from dependencies import SessionDep, get_label_service
from exceptions.custom_errors import ResourceAlreadyExistsError, ResourceNotFoundError, ResourceType
from exceptions.handler import custom_exception_handler
from routers import projects_router
from services.schemas.base import Pagination
from services.schemas.label import LabelSchema, LabelsListSchema


@pytest.fixture
def app():
    from rest.endpoints import labels as _  # noqa: F401

    app = FastAPI()
    app.include_router(projects_router, prefix="/api/v1")
    app.dependency_overrides[SessionDep] = lambda: object()

    # Register the global exception handler
    app.add_exception_handler(Exception, custom_exception_handler)
    app.add_exception_handler(RequestValidationError, custom_exception_handler)

    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def project_id():
    return uuid4()


@pytest.fixture
def label_id():
    return uuid4()


@pytest.fixture
def black_color():
    return Color("#000000")


@pytest.fixture
def red_color():
    return Color("#FF0000")


class TestCreateLabel:
    def test_create_label_success(self, client, project_id, label_id, black_color):
        class FakeLabelService:
            def __init__(self, session):
                pass

            def create_label(self, project_id, create_data):
                return LabelSchema(id=label_id, name=create_data.name, color=black_color.as_hex(format="long"))

        client.app.dependency_overrides[get_label_service] = lambda: FakeLabelService(None)

        label_data = {"name": "test_label"}
        response = client.post(f"/api/v1/projects/{project_id}/labels", json=label_data)

        assert response.status_code == status.HTTP_201_CREATED
        assert "Location" in response.headers
        assert f"/projects/{project_id}/labels/{label_id}" in response.headers["Location"]
        assert response.json()["id"] == str(label_id)
        assert response.json()["name"] == "test_label"
        assert response.json()["color"] == black_color.as_hex(format="long")

    def test_create_label_already_exists(self, client, project_id):
        class FakeLabelService:
            def __init__(self, session):
                pass

            def create_label(self, project_id, create_data):
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.LABEL,
                    resource_value=create_data.name,
                    field="name",
                    message="A label with this name already exists in the project.",
                )

        client.app.dependency_overrides[get_label_service] = lambda: FakeLabelService(None)

        label_data = {"name": "existing_label"}
        response = client.post(f"/api/v1/projects/{project_id}/labels", json=label_data)

        assert response.status_code == status.HTTP_409_CONFLICT
        assert "already exists" in response.json()["detail"].lower()

    def test_create_label_project_not_found(self, client, project_id):
        class FakeLabelService:
            def __init__(self, session):
                pass

            def create_label(self, project_id, create_data):
                raise ResourceNotFoundError(
                    resource_type=ResourceType.PROJECT,
                    resource_id=str(project_id),
                )

        client.app.dependency_overrides[get_label_service] = lambda: FakeLabelService(None)

        label_data = {"name": "test_label"}
        response = client.post(f"/api/v1/projects/{project_id}/labels", json=label_data)

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert str(project_id) in response.json()["detail"]

    def test_create_label_internal_error(self, client, project_id):
        class FakeLabelService:
            def __init__(self, session):
                pass

            def create_label(self, project_id, create_data):
                raise RuntimeError("Database error")

        client.app.dependency_overrides[get_label_service] = lambda: FakeLabelService(None)

        label_data = {"name": "test_label"}
        response = client.post(f"/api/v1/projects/{project_id}/labels", json=label_data)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "internal server error" in response.json()["detail"].lower()


class TestGetLabelById:
    def test_get_label_success(self, client, project_id, label_id, black_color):
        class FakeLabelService:
            def __init__(self, session):
                pass

            def get_label_by_id(self, project_id, label_id):
                return LabelSchema(id=label_id, name="test_label", color=black_color.as_hex(format="long"))

        client.app.dependency_overrides[get_label_service] = lambda: FakeLabelService(None)

        response = client.get(f"/api/v1/projects/{project_id}/labels/{label_id}")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["id"] == str(label_id)
        assert response.json()["name"] == "test_label"
        assert response.json()["color"] == black_color.as_hex(format="long")

    def test_get_label_not_found(self, client, project_id, label_id):
        class FakeLabelService:
            def __init__(self, session):
                pass

            def get_label_by_id(self, project_id, label_id):
                raise ResourceNotFoundError(resource_type=ResourceType.LABEL, resource_id=str(label_id))

        client.app.dependency_overrides[get_label_service] = lambda: FakeLabelService(None)

        response = client.get(f"/api/v1/projects/{project_id}/labels/{label_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_label_internal_error(self, client, project_id, label_id):
        class FakeLabelService:
            def __init__(self, session):
                pass

            def get_label_by_id(self, project_id, label_id):
                raise RuntimeError("Database error")

        client.app.dependency_overrides[get_label_service] = lambda: FakeLabelService(None)

        response = client.get(f"/api/v1/projects/{project_id}/labels/{label_id}")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestGetAllLabels:
    def test_get_all_labels_success(self, client, project_id, black_color, red_color):
        class FakeLabelService:
            def __init__(self, session):
                pass

            def get_all_labels(self, project_id, offset=0, limit=20):
                label_1 = LabelSchema(id=uuid4(), name="test_label", color=black_color.as_hex(format="long"))
                label_2 = LabelSchema(id=uuid4(), name="test_label_2", color=red_color.as_hex(format="long"))
                return LabelsListSchema(
                    labels=[label_1, label_2],
                    pagination=Pagination(count=2, total=2, offset=offset, limit=limit),
                )

        client.app.dependency_overrides[get_label_service] = lambda: FakeLabelService(None)

        response = client.get(f"/api/v1/projects/{project_id}/labels")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()["labels"]) == 2
        assert "pagination" in response.json()

    def test_get_all_labels_empty_list(self, client, project_id):
        class FakeLabelService:
            def __init__(self, session):
                pass

            def get_all_labels(self, project_id, offset=0, limit=20):
                return LabelsListSchema(
                    labels=[],
                    pagination=Pagination(count=0, total=0, offset=offset, limit=limit),
                )

        client.app.dependency_overrides[get_label_service] = lambda: FakeLabelService(None)

        response = client.get(f"/api/v1/projects/{project_id}/labels")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()["labels"]) == 0
        assert "pagination" in response.json()

    def test_get_all_labels_with_pagination(self, client, project_id):
        class FakeLabelService:
            def __init__(self, session):
                pass

            def get_all_labels(self, project_id, offset=0, limit=20):
                return LabelsListSchema(
                    labels=[],
                    pagination=Pagination(count=0, total=0, offset=offset, limit=limit),
                )

        client.app.dependency_overrides[get_label_service] = lambda: FakeLabelService(None)

        response = client.get(f"/api/v1/projects/{project_id}/labels?offset=10&limit=50")

        assert response.status_code == status.HTTP_200_OK

    def test_get_all_labels_internal_error(self, client, project_id):
        class FakeLabelService:
            def __init__(self, session):
                pass

            def get_all_labels(self, project_id, offset=0, limit=20):
                raise RuntimeError("Database error")

        client.app.dependency_overrides[get_label_service] = lambda: FakeLabelService(None)

        response = client.get(f"/api/v1/projects/{project_id}/labels")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestDeleteLabelById:
    def test_delete_label_success(self, client, project_id, label_id):
        class FakeLabelService:
            def __init__(self, session):
                pass

            def delete_label(self, project_id, label_id):
                pass

        client.app.dependency_overrides[get_label_service] = lambda: FakeLabelService(None)

        response = client.delete(f"/api/v1/projects/{project_id}/labels/{label_id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT

    def test_delete_label_not_found(self, client, project_id, label_id):
        class FakeLabelService:
            def __init__(self, session):
                pass

            def delete_label(self, project_id, label_id):
                raise ResourceNotFoundError(resource_type=ResourceType.LABEL, resource_id=str(label_id))

        client.app.dependency_overrides[get_label_service] = lambda: FakeLabelService(None)

        response = client.delete(f"/api/v1/projects/{project_id}/labels/{label_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_label_internal_error(self, client, project_id, label_id):
        class FakeLabelService:
            def __init__(self, session):
                pass

            def delete_label(self, project_id, label_id):
                raise RuntimeError("Database error")

        client.app.dependency_overrides[get_label_service] = lambda: FakeLabelService(None)

        response = client.delete(f"/api/v1/projects/{project_id}/labels/{label_id}")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestUpdateLabel:
    def test_update_label_name_success(self, client, project_id, label_id, black_color):
        class FakeLabelService:
            def __init__(self, session):
                pass

            def update_label(self, project_id, label_id, update_data):
                return LabelSchema(id=label_id, name=update_data.name, color=black_color.as_hex(format="long"))

        client.app.dependency_overrides[get_label_service] = lambda: FakeLabelService(None)

        update_data = {"name": "updated_label"}
        response = client.put(f"/api/v1/projects/{project_id}/labels/{label_id}", json=update_data)

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["id"] == str(label_id)
        assert response.json()["name"] == "updated_label"

    def test_update_label_color_success(self, client, project_id, label_id, red_color):
        class FakeLabelService:
            def __init__(self, session):
                pass

            def update_label(self, project_id, label_id, update_data):
                return LabelSchema(id=label_id, name="test_label", color=update_data.color.as_hex(format="long"))

        client.app.dependency_overrides[get_label_service] = lambda: FakeLabelService(None)

        update_data = {"color": red_color.as_hex(format="long")}
        response = client.put(f"/api/v1/projects/{project_id}/labels/{label_id}", json=update_data)

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["color"] == red_color.as_hex(format="long")

    def test_update_label_project_not_found(self, client, project_id, label_id):
        class FakeLabelService:
            def __init__(self, session):
                pass

            def update_label(self, project_id, label_id, update_data):
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))

        client.app.dependency_overrides[get_label_service] = lambda: FakeLabelService(None)

        update_data = {"name": "updated_label"}
        response = client.put(f"/api/v1/projects/{project_id}/labels/{label_id}", json=update_data)

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_update_label_label_not_found(self, client, project_id, label_id):
        class FakeLabelService:
            def __init__(self, session):
                pass

            def update_label(self, project_id, label_id, update_data):
                raise ResourceNotFoundError(resource_type=ResourceType.LABEL, resource_id=str(label_id))

        client.app.dependency_overrides[get_label_service] = lambda: FakeLabelService(None)

        update_data = {"name": "updated_label"}
        response = client.put(f"/api/v1/projects/{project_id}/labels/{label_id}", json=update_data)

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_update_label_name_already_exists(self, client, project_id, label_id):
        class FakeLabelService:
            def __init__(self, session):
                pass

            def update_label(self, project_id, label_id, update_data):
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.LABEL,
                    resource_value=update_data.name,
                    field="name",
                    message="A label with this name already exists in the project.",
                )

        client.app.dependency_overrides[get_label_service] = lambda: FakeLabelService(None)

        update_data = {"name": "existing_label"}
        response = client.put(f"/api/v1/projects/{project_id}/labels/{label_id}", json=update_data)

        assert response.status_code == status.HTTP_409_CONFLICT
        assert "already exists" in response.json()["detail"].lower()

    def test_update_label_internal_error(self, client, project_id, label_id):
        class FakeLabelService:
            def __init__(self, session):
                pass

            def update_label(self, project_id, label_id, update_data):
                raise RuntimeError("Database error")

        client.app.dependency_overrides[get_label_service] = lambda: FakeLabelService(None)

        update_data = {"name": "updated_label"}
        response = client.put(f"/api/v1/projects/{project_id}/labels/{label_id}", json=update_data)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "internal server error" in response.json()["detail"].lower()
