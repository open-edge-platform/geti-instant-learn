# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from api.endpoints.model_status import _model_status_stream, _scoped_snapshot
from api.error_handler import custom_exception_handler
from dependencies import get_pipeline_manager, get_project_service
from domain.services.schemas.model_status import ModelState, ModelStatusSchema


@pytest.fixture
def project_id():
    return uuid4()


@pytest.fixture
def fake_pipeline_manager():
    mgr = Mock()
    mgr.get_status.return_value = ModelStatusSchema.idle()
    return mgr


@pytest.fixture
def fake_project_service():
    svc = Mock()
    svc.get_project.return_value = Mock()
    return svc


@pytest.fixture
def app(fake_pipeline_manager, fake_project_service):
    from api.endpoints import model_status as _  # noqa: F401
    from api.routers import projects_router

    test_app = FastAPI()
    test_app.add_exception_handler(Exception, custom_exception_handler)
    test_app.add_exception_handler(RequestValidationError, custom_exception_handler)
    test_app.include_router(projects_router, prefix="/api/v1")
    test_app.dependency_overrides[get_pipeline_manager] = lambda: fake_pipeline_manager
    test_app.dependency_overrides[get_project_service] = lambda: fake_project_service
    return test_app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestScopedSnapshot:
    def test_returns_snapshot_when_project_matches(self, project_id):
        snap = ModelStatusSchema.ready(project_id=project_id, model_name="sam3", device="cpu")
        result = _scoped_snapshot(snap, project_id)
        assert result is snap

    def test_returns_idle_when_project_mismatched(self, project_id):
        snap = ModelStatusSchema.ready(project_id=uuid4(), model_name="sam3", device="cpu")
        result = _scoped_snapshot(snap, project_id)
        assert result.state == "idle"
        assert result.project_id == project_id


class TestGetModelStatus:
    def test_snapshot_returns_idle_when_no_pipeline(self, client, project_id, fake_pipeline_manager):
        fake_pipeline_manager.get_status.return_value = ModelStatusSchema.idle()
        response = client.get(f"/api/v1/projects/{project_id}/model-status")
        assert response.status_code == status.HTTP_200_OK
        body = response.json()
        assert body["state"] == "idle"
        assert body["project_id"] == str(project_id)

    def test_snapshot_returns_ready_for_active_project(self, client, project_id, fake_pipeline_manager):
        fake_pipeline_manager.get_status.return_value = ModelStatusSchema.ready(
            project_id=project_id, model_name="sam3", device="cpu"
        )
        response = client.get(f"/api/v1/projects/{project_id}/model-status")
        assert response.status_code == status.HTTP_200_OK
        body = response.json()
        assert body["state"] == "ready"
        assert body["model_name"] == "sam3"
        assert body["device"] == "cpu"

    def test_snapshot_filters_other_project_to_idle(self, client, project_id, fake_pipeline_manager):
        fake_pipeline_manager.get_status.return_value = ModelStatusSchema.ready(
            project_id=uuid4(), model_name="sam3", device="cpu"
        )
        response = client.get(f"/api/v1/projects/{project_id}/model-status")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["state"] == "idle"


class TestModelStatusStreamGenerator:
    """Drive the SSE generator directly to keep tests fast and deterministic."""

    @pytest.mark.asyncio
    async def test_initial_snapshot_then_pushed_event(self, project_id, fake_pipeline_manager):
        queue: asyncio.Queue[ModelStatusSchema] = asyncio.Queue()
        fake_pipeline_manager.subscribe_status.return_value = queue
        fake_pipeline_manager.get_status.return_value = ModelStatusSchema.idle(project_id=project_id)
        loading = ModelStatusSchema.loading_model(project_id=project_id, model_name="sam3", device="cpu")
        queue.put_nowait(loading)
        gen = _model_status_stream(fake_pipeline_manager, project_id)
        first = await gen.__anext__()
        assert first.state == ModelState.IDLE
        assert first.project_id == project_id
        second = await gen.__anext__()
        assert second.state == ModelState.LOADING_MODEL
        assert second.model_name == "sam3"
        assert second.device == "cpu"
        await gen.aclose()
        fake_pipeline_manager.unsubscribe_status.assert_called_once_with(queue)

    @pytest.mark.asyncio
    async def test_unsubscribes_on_close(self, project_id, fake_pipeline_manager):
        queue: asyncio.Queue[ModelStatusSchema] = asyncio.Queue()
        fake_pipeline_manager.subscribe_status.return_value = queue
        fake_pipeline_manager.get_status.return_value = ModelStatusSchema.idle()
        gen = _model_status_stream(fake_pipeline_manager, project_id)
        await gen.__anext__()  # consume initial snapshot to enter the try block
        await gen.aclose()
        fake_pipeline_manager.unsubscribe_status.assert_called_once_with(queue)
