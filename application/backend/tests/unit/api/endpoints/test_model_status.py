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
from domain.errors import ResourceNotFoundError, ResourceType
from domain.services.schemas.model_status import ModelStatus, ModelStatusErrorType, ModelStatusSchema


@pytest.fixture
def project_id():
    return uuid4()


@pytest.fixture
def mock_pipeline_manager():
    mgr = Mock()
    mgr.get_model_status.return_value = ModelStatusSchema(status=ModelStatus.READY)
    return mgr


@pytest.fixture
def mock_project_service():
    svc = Mock()
    svc.get_project.return_value = Mock()
    return svc


@pytest.fixture
def app(mock_pipeline_manager, mock_project_service):
    from api.endpoints import model_status as _  # noqa: F401

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


class TestGetModelStatusEndpoint:
    def test_returns_ready_status_when_model_is_idle(self, client, project_id, mock_pipeline_manager):
        mock_pipeline_manager.get_model_status.return_value = ModelStatusSchema(status=ModelStatus.READY)

        response = client.get(f"/api/v1/projects/{project_id}/model-status")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"status": "ready", "error_type": None, "error_message": None}

    def test_returns_loading_status_when_model_is_busy(self, client, project_id, mock_pipeline_manager):
        mock_pipeline_manager.get_model_status.return_value = ModelStatusSchema(status=ModelStatus.LOADING)

        response = client.get(f"/api/v1/projects/{project_id}/model-status")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"status": "loading", "error_type": None, "error_message": None}

    def test_returns_error_status_when_last_model_load_failed(self, client, project_id, mock_pipeline_manager):
        error_message = (
            "User does not have access to the weights of the DinoV3 model.\n"
            "Please follow these steps:\n"
            "1. Request access on the HuggingFace website: "
            "https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m\n"
            "2. Set your HuggingFace credentials using one of these methods:\n"
            "   - Run: hf auth login\n"
            "   - Set environment variable: export HUGGINGFACE_HUB_TOKEN=your_token"
        )
        mock_pipeline_manager.get_model_status.return_value = ModelStatusSchema(
            status=ModelStatus.ERROR,
            error_type=ModelStatusErrorType.ACCESS_REQUIRED,
            error_message=error_message,
        )

        response = client.get(f"/api/v1/projects/{project_id}/model-status")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "status": "error",
            "error_type": "access_required",
            "error_message": error_message,
        }

    def test_returns_404_when_project_not_found(self, client, project_id, mock_project_service):
        mock_project_service.get_project.side_effect = ResourceNotFoundError(
            ResourceType.PROJECT, str(project_id), "Project not found"
        )

        response = client.get(f"/api/v1/projects/{project_id}/model-status")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()

    def test_returns_400_for_invalid_project_id(self, client):
        response = client.get("/api/v1/projects/not-a-uuid/model-status")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
