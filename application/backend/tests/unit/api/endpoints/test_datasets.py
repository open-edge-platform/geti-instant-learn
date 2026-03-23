# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from uuid import uuid4, uuid5

from fastapi import FastAPI
from fastapi.testclient import TestClient

from domain.services.schemas.base import Pagination
from domain.services.schemas.dataset import DatasetSchema, DatasetsListSchema
from runtime.services.dataset_discovery import DATASET_NS, scan_datasets


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
        ],
        pagination=Pagination(count=2, total=2, offset=0, limit=2),
    )
    response = _create_client(response_payload).get("/api/v1/system/datasets")

    assert response.status_code == 200
    body = response.json()
    assert "datasets" in body
    assert len(body["datasets"]) == 2
    names = {dataset["name"] for dataset in body["datasets"]}
    assert names == {"Aquarium", "Nuts"}
    assert body["pagination"] == {"count": 2, "total": 2, "offset": 0, "limit": 20}


def test_get_datasets_empty_list_when_cache_is_empty():
    response = _create_client(
        DatasetsListSchema(
            datasets=[],
            pagination=Pagination(count=0, total=0, offset=0, limit=0),
        )
    ).get("/api/v1/system/datasets")

    assert response.status_code == 200
    assert response.json() == {
        "datasets": [],
        "pagination": {"count": 0, "total": 0, "offset": 0, "limit": 20},
    }


def test_get_datasets_with_offset_and_limit(tmp_path: Path):
    response_payload = DatasetsListSchema(
        datasets=[
            DatasetSchema(id=uuid4(), name="Aquarium", description="This is sample dataset of aquarium."),
            DatasetSchema(id=uuid4(), name="Nuts", description="This is sample dataset of nuts."),
            DatasetSchema(id=uuid4(), name="Potatoes", description="This is sample dataset of potatoes."),
        ],
        pagination=Pagination(count=3, total=3, offset=0, limit=3),
    )

    response = _create_client(response_payload).get("/api/v1/system/datasets?offset=1&limit=1")

    assert response.status_code == 200
    body = response.json()
    assert len(body["datasets"]) == 1
    assert body["datasets"][0]["name"] == "Nuts"
    assert body["pagination"] == {"count": 1, "total": 3, "offset": 1, "limit": 1}


def test_scan_datasets_builds_id_to_path_mapping(tmp_path: Path):
    dataset_dir = tmp_path / "aquarium"
    dataset_dir.mkdir()

    datasets, dataset_paths = scan_datasets(tmp_path)

    assert len(datasets.datasets) == 1
    assert datasets.pagination.count == 1
    assert datasets.pagination.total == 1
    dataset_id = datasets.datasets[0].id
    assert dataset_id == uuid5(DATASET_NS, "aquarium")
    assert dataset_id in dataset_paths
    assert dataset_paths[dataset_id] == dataset_dir
