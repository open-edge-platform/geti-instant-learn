# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from api.error_handler import custom_exception_handler
from api.routers import projects_router
from dependencies import get_pipeline_manager, get_project_service
from domain.errors import ResourceNotFoundError, ResourceType, ResourceUpdateConflictError
from runtime.errors import PipelineReloadInProgressError


@pytest.fixture
def project_id():
    return uuid4()


@pytest.fixture
def mock_pipeline_manager():
    return Mock()


@pytest.fixture
def mock_project_service():
    svc = Mock()
    svc.ensure_project_is_active.return_value = None
    return svc


@pytest.fixture
def app(mock_pipeline_manager, mock_project_service):
    from api.endpoints import projects as _  # noqa: F401

    test_app = FastAPI()
    test_app.add_exception_handler(Exception, custom_exception_handler)
    test_app.add_exception_handler(RequestValidationError, custom_exception_handler)
    test_app.include_router(projects_router, prefix="/api/v1")
    test_app.dependency_overrides[get_pipeline_manager] = lambda: mock_pipeline_manager
    test_app.dependency_overrides[get_project_service] = lambda: mock_project_service
    return test_app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestReloadProjectPipelineEndpoint:
    def test_returns_202_and_reloads_pipeline(self, client, project_id, mock_pipeline_manager):
        response = client.post(f"/api/v1/projects/{project_id}/reload")

        assert response.status_code == status.HTTP_202_ACCEPTED
        mock_pipeline_manager.reload_pipeline.assert_called_once_with(project_id)

    def test_returns_404_when_project_not_found(self, client, project_id, mock_project_service):
        mock_project_service.ensure_project_is_active.side_effect = ResourceNotFoundError(
            ResourceType.PROJECT, str(project_id), "Project not found"
        )

        response = client.post(f"/api/v1/projects/{project_id}/reload")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_returns_400_when_project_is_not_active(self, client, project_id, mock_project_service):
        mock_project_service.ensure_project_is_active.side_effect = ResourceUpdateConflictError(
            resource_type=ResourceType.PROJECT,
            resource_id=str(project_id),
            field="active",
            message="Only the active project pipeline can be reloaded.",
        )

        response = client.post(f"/api/v1/projects/{project_id}/reload")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "active project" in response.json()["detail"].lower()
        mock_pipeline_manager = client.app.dependency_overrides[get_pipeline_manager]()
        mock_pipeline_manager.reload_pipeline.assert_not_called()

    def test_returns_409_when_reload_is_already_in_progress(self, client, project_id, mock_pipeline_manager):
        mock_pipeline_manager.reload_pipeline.side_effect = PipelineReloadInProgressError(
            "Pipeline reload is already in progress."
        )

        response = client.post(f"/api/v1/projects/{project_id}/reload")

        assert response.status_code == status.HTTP_409_CONFLICT
        assert "already in progress" in response.json()["detail"]
