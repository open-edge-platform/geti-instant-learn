# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from uuid import uuid4, uuid5

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.error_handler import custom_exception_handler
from domain.services.schemas.base import Pagination
from domain.services.schemas.dataset import DatasetSchema, DatasetsListSchema
from runtime.services.dataset_discovery import DATASET_NS, scan_datasets


def _create_client(datasets: DatasetsListSchema) -> TestClient:
    app = FastAPI()
    app.add_exception_handler(Exception, custom_exception_handler)
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

    assert response.status_code == 404
    assert response.json() == {"detail": "No datasets found in startup cache."}


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


def test_scan_datasets_sets_thumbnail_from_first_supported_image(tmp_path: Path):
    dataset_dir = tmp_path / "aquarium"
    dataset_dir.mkdir()

    # Non-image files should be ignored when selecting thumbnail source.
    (dataset_dir / "README.txt").write_text("info")

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, :] = [255, 0, 0]
    cv2.imwrite(str(dataset_dir / "0001.jpg"), image)

    datasets, _ = scan_datasets(tmp_path)

    assert len(datasets.datasets) == 1
    thumbnail = datasets.datasets[0].thumbnail
    assert thumbnail is not None
    assert thumbnail.startswith("data:image/jpeg;base64,")


def test_scan_datasets_thumbnail_none_when_no_supported_images(tmp_path: Path):
    dataset_dir = tmp_path / "aquarium"
    dataset_dir.mkdir()
    (dataset_dir / "README.txt").write_text("no images")

    datasets, _ = scan_datasets(tmp_path)

    assert len(datasets.datasets) == 1
    assert datasets.datasets[0].thumbnail is None
