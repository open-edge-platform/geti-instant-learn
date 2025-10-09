# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.components.schemas.reader import SourceType, WebCamConfig
from dependencies import SessionDep
from routers import projects_router
from services.errors import (
    ResourceNotFoundError,
    ResourceType,
    ResourceUpdateConflictError,
)
from services.schemas.source import SourceSchema, SourcesListSchema

PROJECT_ID = uuid4()
SOURCE_ID_1 = uuid4()
SOURCE_ID_2 = uuid4()


def make_source_schema(
    source_id: UUID,
    device_id: int,
    connected: bool = False,
) -> SourceSchema:
    return SourceSchema(
        id=source_id,
        connected=connected,
        config=WebCamConfig(source_type=SourceType.WEBCAM, device_id=device_id),
    )


@pytest.fixture
def app():
    # ensure sources endpoints register routes on projects_router before inclusion.
    from rest.endpoints import sources as _  # noqa: F401

    app = FastAPI()
    app.include_router(projects_router, prefix="/api/v1")
    app.dependency_overrides[SessionDep] = lambda: object()
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.mark.parametrize(
    "behavior,expected_status,expected_len,expected_detail",
    [
        ("some", 200, 2, None),
        ("notfound", 404, None, None),
        ("error", 500, None, "Failed to list sources."),
    ],
)
def test_get_sources(client, monkeypatch, behavior, expected_status, expected_len, expected_detail):
    from rest.endpoints import sources as ep_mod

    class FakeService:
        def __init__(self, session):
            pass

        def list_sources(self, project_id: UUID):
            assert project_id == PROJECT_ID
            if behavior == "some":
                return SourcesListSchema(
                    sources=[
                        make_source_schema(SOURCE_ID_1, 0, True),
                        make_source_schema(SOURCE_ID_2, 1, False),
                    ]
                )
            if behavior == "notfound":
                raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(ep_mod, "SourceService", FakeService)

    resp = client.get(f"/api/v1/projects/{PROJECT_ID}/sources")
    assert resp.status_code == expected_status
    if expected_detail:
        assert resp.json()["detail"] == expected_detail
        return
    if behavior == "notfound":
        assert "detail" in resp.json()
        return
    data = resp.json()
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) == expected_len
    if behavior == "some":
        ids = {s["id"] for s in data["sources"]}
        assert ids == {str(SOURCE_ID_1), str(SOURCE_ID_2)}
        first = data["sources"][0]
        assert "config" in first
        assert first["config"]["source_type"] == "webcam"
        assert "device_id" in first["config"]


@pytest.mark.parametrize(
    "behavior,expected_status,expect_detail",
    [
        ("success", 201, None),
        ("conflict", 409, None),
        ("notfound", 404, None),
        ("error", 500, "Failed to create source."),
    ],
)
def test_create_source(client, monkeypatch, behavior, expected_status, expect_detail):
    from rest.endpoints import sources as ep_mod

    CREATED_ID = uuid4()

    class FakeService:
        def __init__(self, session):
            pass

        def create_source(self, project_id: UUID, create_data):
            assert project_id == PROJECT_ID
            assert create_data.config.source_type == SourceType.WEBCAM
            if behavior == "success":
                return make_source_schema(CREATED_ID, create_data.config.device_id, create_data.connected)
            if behavior == "conflict":
                raise ResourceUpdateConflictError(ResourceType.SOURCE, str(CREATED_ID), field="source_type")
            if behavior == "notfound":
                raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(ep_mod, "SourceService", FakeService)

    payload = {
        "id": str(CREATED_ID),
        "connected": True,
        "config": {"source_type": "webcam", "device_id": 3},
    }
    resp = client.post(f"/api/v1/projects/{PROJECT_ID}/sources", json=payload)
    assert resp.status_code == expected_status
    if expect_detail:
        assert resp.json()["detail"] == expect_detail
        return
    if behavior in ("conflict", "notfound"):
        assert "detail" in resp.json()
        return
    data = resp.json()
    assert data["id"] == str(CREATED_ID)
    assert data["connected"] is True
    assert data["config"]["source_type"] == "webcam"
    assert data["config"]["device_id"] == 3


@pytest.mark.parametrize(
    "behavior,expected_status,expect_detail",
    [
        ("success", 200, None),
        ("conflict", 409, None),
        ("notfound", 404, None),
        ("error", 500, "Failed to update source configuration."),
    ],
)
def test_update_source(client, monkeypatch, behavior, expected_status, expect_detail):
    from rest.endpoints import sources as ep_mod

    class FakeService:
        def __init__(self, session):
            pass

        def update_source(self, project_id: UUID, source_id: UUID, update_data):
            assert project_id == PROJECT_ID
            assert source_id == SOURCE_ID_1
            assert update_data.config.source_type == SourceType.WEBCAM
            if behavior == "success":
                return make_source_schema(source_id, update_data.config.device_id, update_data.connected)
            if behavior == "conflict":
                raise ResourceUpdateConflictError(ResourceType.SOURCE, str(source_id), field="source_type")
            if behavior == "notfound":
                raise ResourceNotFoundError(ResourceType.SOURCE, str(source_id))
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(ep_mod, "SourceService", FakeService)

    payload = {
        "connected": False,
        "config": {"source_type": "webcam", "device_id": 7},
    }
    resp = client.put(f"/api/v1/projects/{PROJECT_ID}/sources/{SOURCE_ID_1}", json=payload)
    assert resp.status_code == expected_status
    if expect_detail:
        assert resp.json()["detail"] == expect_detail
        return
    if behavior in ("conflict", "notfound"):
        assert "detail" in resp.json()
        return
    data = resp.json()
    assert data["id"] == str(SOURCE_ID_1)
    assert data["config"]["device_id"] == 7
    assert data["config"]["source_type"] == "webcam"


@pytest.mark.parametrize(
    "behavior,expected_status,expect_detail",
    [
        ("success", 204, None),
        ("missing", 204, None),
        ("error", 500, "Failed to delete source."),
    ],
)
def test_delete_source(client, monkeypatch, behavior, expected_status, expect_detail):
    from rest.endpoints import sources as ep_mod

    class FakeService:
        def __init__(self, session):
            pass

        def delete_source(self, project_id: UUID, source_id: UUID):
            assert project_id == PROJECT_ID
            assert source_id == SOURCE_ID_2
            if behavior == "success":
                return
            if behavior == "missing":
                raise ResourceNotFoundError(ResourceType.SOURCE, str(source_id))
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(ep_mod, "SourceService", FakeService)

    resp = client.delete(f"/api/v1/projects/{PROJECT_ID}/sources/{SOURCE_ID_2}")
    assert resp.status_code == expected_status
    if expect_detail:
        assert resp.json()["detail"] == expect_detail
    else:
        if resp.status_code == 204:
            assert resp.text == ""
