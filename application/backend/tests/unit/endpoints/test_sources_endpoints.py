# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.components.schemas.reader import SourceType
from dependencies import SessionDep  # type: ignore
from routers import projects_router
from services.common import (
    ResourceNotFoundError,
    ResourceType,
    ResourceUpdateConflictError,
)
from services.schemas.source import SourceSchema, SourcesListSchema

PROJECT_ID = uuid4()
PROJECT_ID_STR = str(PROJECT_ID)
SOURCE_ID = uuid4()
SOURCE_ID_2 = uuid4()


def make_source_schema(source_id: UUID = SOURCE_ID, name: str = "cam0", device_id: int = 0) -> SourceSchema:
    return SourceSchema(
        source_type=SourceType.WEBCAM,
        id=source_id,
        name=name,
        device_id=device_id,
        connected=False,
    )


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
    "behavior,expected_status,expected_count,expected_detail",
    [
        ("empty", 200, 0, None),
        ("some", 200, 2, None),
        ("notfound", 404, None, None),
        ("error", 500, None, "Failed to list sources."),
    ],
)
def test_get_sources(client, monkeypatch, behavior, expected_status, expected_count, expected_detail):
    from rest.endpoints import sources as ep_mod

    class FakeService:
        def __init__(self, session):
            pass

        def list_sources(self, project_id: UUID):
            assert project_id == PROJECT_ID
            if behavior == "empty":
                return SourcesListSchema(sources=[])
            if behavior == "some":
                return SourcesListSchema(
                    sources=[
                        make_source_schema(SOURCE_ID, "cam0", 0),
                        make_source_schema(SOURCE_ID_2, "cam1", 1),
                    ]
                )
            if behavior == "notfound":
                raise ResourceNotFoundError(
                    resource_type=ResourceType.PROJECT,
                    resource_id=str(project_id),
                )
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(ep_mod, "SourceService", FakeService)
    resp = client.get(f"/api/v1/projects/{PROJECT_ID_STR}/sources")

    assert resp.status_code == expected_status
    if expected_detail:
        assert resp.json()["detail"] == expected_detail
        return
    if behavior == "notfound":
        assert "detail" in resp.json()
        return
    data = resp.json()
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) == expected_count
    if behavior == "some":
        ids = {s["id"] for s in data["sources"]}
        assert ids == {str(SOURCE_ID), str(SOURCE_ID_2)}
        first = data["sources"][0]
        assert first["source_type"] == "webcam"
        assert "device_id" in first


@pytest.mark.parametrize(
    "behavior,expected_status",
    [
        ("create", 201),
        ("update", 200),
        ("conflict", 409),
        ("notfound", 404),
        ("error", 500),
    ],
)
def test_update_source(client, monkeypatch, behavior, expected_status):  # noqa C901
    from rest.endpoints import sources as ep_mod

    CREATED_NAME = "newcam"
    UPDATED_NAME = "updatedcam"

    class FakeService:
        def __init__(self, session):
            pass

        def upsert_source(self, project_id: UUID, source_id: UUID, payload):
            assert project_id == PROJECT_ID
            assert source_id == SOURCE_ID
            assert payload.source_type == SourceType.WEBCAM
            if behavior == "create":
                return make_source_schema(SOURCE_ID, CREATED_NAME, payload.device_id), True
            if behavior == "update":
                return make_source_schema(SOURCE_ID, UPDATED_NAME, payload.device_id), False
            if behavior == "conflict":
                raise ResourceUpdateConflictError(
                    resource_type=ResourceType.SOURCE,
                    resource_id=str(source_id),
                    field="source_type",
                )
            if behavior == "notfound":
                raise ResourceNotFoundError(
                    resource_type=ResourceType.PROJECT,
                    resource_id=str(project_id),
                )
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(ep_mod, "SourceService", FakeService)

    payload = {
        "source_type": "webcam",
        "name": "some name",
        "device_id": 5,
    }
    resp = client.put(f"/api/v1/projects/{PROJECT_ID_STR}/sources/{SOURCE_ID}", json=payload)
    assert resp.status_code == expected_status

    if behavior in ("create", "update"):
        data = resp.json()
        assert data["id"] == str(SOURCE_ID)
        assert data["source_type"] == "webcam"
        assert data["device_id"] == 5
        if behavior == "create":
            assert data["name"] == CREATED_NAME
        else:
            assert data["name"] == UPDATED_NAME
    elif behavior == "conflict":
        assert "detail" in resp.json()
    elif behavior == "notfound":
        assert "detail" in resp.json()
    elif behavior == "error":
        assert resp.json()["detail"] == "Failed to update source configuration."


@pytest.mark.parametrize(
    "behavior,expected_status,expect_detail",
    [
        ("success", 204, None),
        ("missing", 204, None),  # suppressed
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
            assert source_id == SOURCE_ID
            if behavior == "success":
                return
            if behavior == "missing":
                raise ResourceNotFoundError(
                    resource_type=ResourceType.SOURCE,
                    resource_id=str(source_id),
                )
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(ep_mod, "SourceService", FakeService)

    resp = client.delete(f"/api/v1/projects/{PROJECT_ID_STR}/sources/{SOURCE_ID}")
    assert resp.status_code == expected_status
    if expect_detail:
        assert resp.json()["detail"] == expect_detail
    else:
        assert resp.text == ""
