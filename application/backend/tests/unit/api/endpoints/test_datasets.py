# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from uuid import uuid4, uuid5

from fastapi import FastAPI
from fastapi.testclient import TestClient

from domain.services.dataset_discovery import DATASET_NS, scan_datasets
from domain.services.schemas.dataset import DatasetSchema, DatasetsListSchema


def _create_client(datasets: DatasetsListSchema) -> TestClient:
    app = FastAPI()
    app.state.available_datasets = datasets
    app.state.dataset_paths = {}

    from api.endpoints import datasets as _  # noqa: F401
    from api.routers import system_router

    app.include_router(system_router, prefix="/api/v1")
    return TestClient(app, raise_server_exceptions=False)


def test_get_datasets_success(tmp_path: Path):
    response_payload = DatasetsListSchema(
        datasets=[
            DatasetSchema(id=uuid4(), name="Aquarium", description="This is sample dataset of aquarium."),
            DatasetSchema(id=uuid4(), name="Nuts", description="This is sample dataset of nuts."),
        ]
    )
    response = _create_client(response_payload).get("/api/v1/system/datasets")

    assert response.status_code == 200
    body = response.json()
    assert "datasets" in body
    assert len(body["datasets"]) == 2
    names = {dataset["name"] for dataset in body["datasets"]}
    assert names == {"Aquarium", "Nuts"}


def test_get_datasets_empty_list_when_cache_is_empty():
    response = _create_client(DatasetsListSchema(datasets=[])).get("/api/v1/system/datasets")

    assert response.status_code == 200
    assert response.json() == {"datasets": []}


def test_scan_datasets_builds_id_to_path_mapping(tmp_path: Path):
    dataset_dir = tmp_path / "aquarium"
    dataset_dir.mkdir()

    datasets, dataset_paths = scan_datasets(tmp_path)

    assert len(datasets.datasets) == 1
    dataset_id = datasets.datasets[0].id
    assert dataset_id == uuid5(DATASET_NS, "aquarium")
    assert dataset_id in dataset_paths
    assert dataset_paths[dataset_id] == dataset_dir
