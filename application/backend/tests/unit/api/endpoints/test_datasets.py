# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.error_handler import custom_exception_handler
from dependencies import get_dataset_registry_service
from domain.services.schemas.dataset import DatasetSchema, DatasetsListSchema


@pytest.fixture
def mock_dataset_registry_service() -> MagicMock:
    return MagicMock()


@pytest.fixture
def app(mock_dataset_registry_service: MagicMock) -> FastAPI:
    app = FastAPI()
    app.add_exception_handler(Exception, custom_exception_handler)
    app.dependency_overrides[get_dataset_registry_service] = lambda: mock_dataset_registry_service

    from api.endpoints import datasets as _  # noqa: F401
    from api.routers import system_router

    app.include_router(system_router, prefix="/api/v1")
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def test_get_datasets_success(client: TestClient, mock_dataset_registry_service: MagicMock):
    dataset = DatasetSchema(id=uuid4(), name="Aquarium", description="This is sample dataset of aquarium.")
    mock_dataset_registry_service.list_datasets.return_value = DatasetsListSchema(datasets=[dataset])

    response = client.get("/api/v1/system/datasets")

    assert response.status_code == 200
    body = response.json()
    assert "datasets" in body
    assert len(body["datasets"]) == 1
    assert body["datasets"][0]["id"] == str(dataset.id)
    assert body["datasets"][0]["name"] == "Aquarium"
    assert body["datasets"][0]["description"] == "This is sample dataset of aquarium."
    mock_dataset_registry_service.list_datasets.assert_called_once_with()


def test_get_datasets_error(client: TestClient, mock_dataset_registry_service: MagicMock):
    mock_dataset_registry_service.list_datasets.side_effect = RuntimeError("Unexpected failure")

    response = client.get("/api/v1/system/datasets")

    assert response.status_code == 500
    assert "internal server error" in response.json()["detail"].lower()
    mock_dataset_registry_service.list_datasets.assert_called_once_with()


def test_get_datasets_empty_list(client: TestClient, mock_dataset_registry_service: MagicMock):
    mock_dataset_registry_service.list_datasets.return_value = DatasetsListSchema(datasets=[])

    response = client.get("/api/v1/system/datasets")

    assert response.status_code == 200
    body = response.json()
    assert body == {"datasets": []}
    mock_dataset_registry_service.list_datasets.assert_called_once_with()
