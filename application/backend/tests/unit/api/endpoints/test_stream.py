# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from api.error_handler import custom_exception_handler
from api.routers import projects_router
from dependencies import get_mjpeg_stream_service, get_pipeline_manager
from runtime.services.mjpeg_stream import BOUNDARY

PROJECT_ID = uuid4()
PROJECT_ID_STR = str(PROJECT_ID)


@pytest.fixture
def mock_pipeline_manager():
    manager = MagicMock()
    manager.get_output_slot.return_value = MagicMock()
    manager.get_visualization_info.return_value = None
    return manager


@pytest.fixture
def mock_mjpeg_service():
    service = MagicMock()

    async def _fake_stream(*args, **kwargs):
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfake-jpeg\r\n"

    service.stream = MagicMock(side_effect=_fake_stream)
    return service


@pytest.fixture
def app(mock_pipeline_manager, mock_mjpeg_service):
    from api.endpoints import stream as _  # noqa: F401

    app = FastAPI()
    app.include_router(projects_router, prefix="/api/v1")

    app.dependency_overrides[get_pipeline_manager] = lambda: mock_pipeline_manager
    app.dependency_overrides[get_mjpeg_stream_service] = lambda: mock_mjpeg_service

    app.add_exception_handler(Exception, custom_exception_handler)
    app.add_exception_handler(RequestValidationError, custom_exception_handler)

    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestStreamMjpegEndpoint:
    def test_stream_returns_200_with_multipart_content_type(self, client):
        response = client.get(f"/api/v1/projects/{PROJECT_ID_STR}/stream")

        assert response.status_code == 200
        assert f"multipart/x-mixed-replace; boundary={BOUNDARY}" in response.headers["content-type"]

    def test_stream_returns_mjpeg_body(self, client):
        response = client.get(f"/api/v1/projects/{PROJECT_ID_STR}/stream")

        assert b"fake-jpeg" in response.content

    def test_stream_calls_pipeline_manager_with_project_id(self, client, mock_pipeline_manager):
        client.get(f"/api/v1/projects/{PROJECT_ID_STR}/stream")

        mock_pipeline_manager.get_output_slot.assert_called_once_with(project_id=PROJECT_ID)

    def test_stream_passes_output_slot_to_service(self, client, mock_pipeline_manager, mock_mjpeg_service):
        client.get(f"/api/v1/projects/{PROJECT_ID_STR}/stream")

        args, _ = mock_mjpeg_service.stream.call_args
        assert args[0] is mock_pipeline_manager.get_output_slot.return_value

    def test_stream_invalid_project_id_returns_400(self, client):
        response = client.get("/api/v1/projects/not-a-uuid/stream")

        assert response.status_code == 400

    def test_stream_pipeline_manager_error_returns_500(self, client, mock_pipeline_manager):
        mock_pipeline_manager.get_output_slot.side_effect = RuntimeError("no pipeline")
        response = client.get(f"/api/v1/projects/{PROJECT_ID_STR}/stream")

        assert response.status_code == 500
