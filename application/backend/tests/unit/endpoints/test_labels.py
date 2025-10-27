# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from pydantic_extra_types.color import Color

from dependencies import SessionDep, get_config_dispatcher
from routers import projects_router
from services.errors import ResourceAlreadyExistsError, ResourceNotFoundError, ResourceType
from services.schemas.base import Pagination
from services.schemas.label import LabelSchema, LabelsListSchema


@pytest.fixture(autouse=True)
def app():
    app = FastAPI()
    app.include_router(projects_router, prefix="/api/v1")
    app.dependency_overrides[SessionDep] = lambda: object()

    class DummyDispatcher:
        def dispatch(self, event):  # noqa: D401
            pass

    app.dependency_overrides[get_config_dispatcher] = lambda: DummyDispatcher()
    return app


@pytest.fixture(autouse=True)
def fxt_client(app):
    return TestClient(app)


@pytest.fixture
def mock_label_service():
    with patch("rest.endpoints.labels.LabelService") as mock:
        yield mock


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
    def test_create_label_success(self, fxt_client, mock_label_service, project_id, label_id, black_color):
        label_data = {"name": "test_label"}
        mock_label = LabelSchema(id=label_id, name="test_label", color=black_color.as_hex(format="long"))
        mock_label_service.return_value.create_label.return_value = mock_label

        response = fxt_client.post(f"/api/v1/projects/{project_id}/labels", json=label_data)

        assert response.status_code == status.HTTP_201_CREATED
        assert "Location" in response.headers
        assert f"/projects/{project_id}/labels/{mock_label.id}" in response.headers["Location"]
        assert response.json()["id"] == str(label_id)
        assert response.json()["name"] == "test_label"
        assert response.json()["color"] == black_color.as_hex(format="long")

    def test_create_label_already_exists(self, fxt_client, mock_label_service, project_id):
        label_data = {"name": "existing_label"}
        mock_label_service.return_value.create_label.side_effect = ResourceAlreadyExistsError(
            resource_type=ResourceType.LABEL,
            resource_value=label_data["name"],
        )

        response = fxt_client.post(f"/api/v1/projects/{project_id}/labels", json=label_data)

        assert response.status_code == status.HTTP_409_CONFLICT
        assert "existing_label" in response.json()["detail"]

    def test_create_label_project_not_found(self, fxt_client, mock_label_service, project_id):
        label_data = {"name": "test_label"}
        mock_label_service.return_value.create_label.side_effect = ResourceNotFoundError(
            resource_type=ResourceType.PROJECT, resource_id=str(project_id)
        )

        response = fxt_client.post(f"/api/v1/projects/{project_id}/labels", json=label_data)

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert f"Project with ID {project_id} not found" in response.json()["detail"]

    def test_create_label_internal_error(self, fxt_client, mock_label_service, project_id):
        label_data = {"name": "test_label"}
        mock_label_service.return_value.create_label.side_effect = Exception("Database error")

        response = fxt_client.post(f"/api/v1/projects/{project_id}/labels", json=label_data)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "internal server error" in response.json()["detail"].lower()


class TestGetLabelById:
    def test_get_label_success(self, fxt_client, mock_label_service, project_id, label_id, black_color):
        mock_label = LabelSchema(id=label_id, name="test_label", color=black_color.as_hex(format="long"))
        mock_label_service.return_value.get_label_by_id.return_value = mock_label

        response = fxt_client.get(f"/api/v1/projects/{project_id}/labels/{label_id}")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["id"] == str(label_id)
        assert response.json()["name"] == "test_label"
        assert response.json()["color"] == black_color.as_hex(format="long")

    def test_get_label_not_found(self, fxt_client, mock_label_service, project_id, label_id):
        mock_label_service.return_value.get_label_by_id.side_effect = ResourceNotFoundError(
            resource_type=ResourceType.LABEL, resource_id=str(label_id)
        )

        response = fxt_client.get(f"/api/v1/projects/{project_id}/labels/{label_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_label_internal_error(self, fxt_client, mock_label_service, project_id, label_id):
        mock_label_service.return_value.get_label_by_id.side_effect = Exception("Database error")

        response = fxt_client.get(f"/api/v1/projects/{project_id}/labels/{label_id}")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestGetAllLabels:
    def test_get_all_labels_success(self, fxt_client, mock_label_service, project_id, black_color, red_color):
        label_1 = LabelSchema(id=uuid4(), name="test_label", color=black_color.as_hex(format="long"))
        label_2 = LabelSchema(id=uuid4(), name="test_label_2", color=red_color.as_hex(format="long"))
        mock_labels = LabelsListSchema(
            labels=[label_1, label_2],
            pagination=Pagination(
                count=0,
                total=0,
                offset=0,
                limit=0,
            ),
        )
        mock_label_service.return_value.get_all_labels.return_value = mock_labels

        response = fxt_client.get(f"/api/v1/projects/{project_id}/labels")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()["labels"]) == 2
        assert "pagination" in response.json()

    def test_get_all_labels_empty_list(self, fxt_client, mock_label_service, project_id):
        mock_labels = LabelsListSchema(
            labels=[],
            pagination=Pagination(
                count=0,
                total=0,
                offset=0,
                limit=0,
            ),
        )
        mock_label_service.return_value.get_all_labels.return_value = mock_labels

        response = fxt_client.get(f"/api/v1/projects/{project_id}/labels")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()["labels"]) == 0
        assert "pagination" in response.json()

    def test_get_all_labels_with_pagination(self, fxt_client, mock_label_service, project_id):
        mock_labels = LabelsListSchema(
            labels=[],
            pagination=Pagination(
                count=0,
                total=0,
                offset=0,
                limit=0,
            ),
        )
        mock_label_service.return_value.get_all_labels.return_value = mock_labels

        response = fxt_client.get(f"/api/v1/projects/{project_id}/labels?offset=10&limit=50")

        assert response.status_code == status.HTTP_200_OK
        mock_label_service.return_value.get_all_labels.assert_called_once_with(
            project_id=project_id, offset=10, limit=50
        )

    def test_get_all_labels_internal_error(self, fxt_client, mock_label_service, project_id):
        mock_label_service.return_value.get_all_labels.side_effect = Exception("Database error")

        response = fxt_client.get(f"/api/v1/projects/{project_id}/labels")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestDeleteLabelById:
    def test_delete_label_success(self, fxt_client, mock_label_service, project_id, label_id):
        mock_label_service.return_value.delete_label.return_value = None

        response = fxt_client.delete(f"/api/v1/projects/{project_id}/labels/{label_id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT

    def test_delete_label_not_found(self, fxt_client, mock_label_service, project_id, label_id):
        mock_label_service.return_value.delete_label.side_effect = ResourceNotFoundError(
            resource_type=ResourceType.LABEL, resource_id=str(label_id)
        )

        response = fxt_client.delete(f"/api/v1/projects/{project_id}/labels/{label_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_label_internal_error(self, fxt_client, mock_label_service, project_id, label_id):
        mock_label_service.return_value.delete_label.side_effect = Exception("Database error")

        response = fxt_client.delete(f"/api/v1/projects/{project_id}/labels/{label_id}")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
