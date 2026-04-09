# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from uuid import uuid4, uuid5

import pytest

from domain.errors import DatasetNotFoundError
from domain.services.dataset_discovery import (
    DATASET_NS,
    _get_first_image,
    get_first_dataset_path,
    resolve_dataset_path,
    scan_datasets,
)


class TestResolveDatasetPath:
    def test_returns_matching_dataset_directory(self, tmp_path: Path) -> None:
        (tmp_path / "zebra").mkdir()
        target_dir = tmp_path / "aquarium"
        target_dir.mkdir()

        dataset_id = uuid5(DATASET_NS, "aquarium")

        resolved_path = resolve_dataset_path(dataset_id, tmp_path)

        assert resolved_path == target_dir

    def test_returns_none_when_dataset_id_does_not_exist(self, tmp_path: Path) -> None:
        (tmp_path / "aquarium").mkdir()

        resolved_path = resolve_dataset_path(uuid4(), tmp_path)

        assert resolved_path is None

    def test_returns_none_when_template_dataset_path_is_missing(self, tmp_path: Path) -> None:
        missing_dir = tmp_path / "missing"

        resolved_path = resolve_dataset_path(uuid4(), missing_dir)

        assert resolved_path is None


class TestGetFirstDatasetPath:
    def test_returns_first_directory_in_sorted_order(self, tmp_path: Path) -> None:
        (tmp_path / "b_dataset").mkdir()
        expected_dir = tmp_path / "a_dataset"
        expected_dir.mkdir()
        (tmp_path / "notes.txt").touch()

        first_path = get_first_dataset_path(tmp_path)

        assert first_path == expected_dir

    def test_returns_none_when_no_dataset_directories_exist(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").touch()

        first_path = get_first_dataset_path(tmp_path)

        assert first_path is None

    def test_returns_none_when_template_dataset_path_is_missing(self, tmp_path: Path) -> None:
        missing_dir = tmp_path / "missing"

        first_path = get_first_dataset_path(missing_dir)

        assert first_path is None


class TestGetFirstImage:
    def test_returns_first_supported_image_in_sorted_order(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("domain.services.dataset_discovery.settings.supported_extensions", {".jpg", ".png"})

        (tmp_path / "z_image.png").touch()
        expected = tmp_path / "a_image.jpg"
        expected.touch()
        (tmp_path / "README.txt").touch()

        first_image = _get_first_image(tmp_path)

        assert first_image == expected

    def test_returns_none_when_no_supported_images_exist(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("domain.services.dataset_discovery.settings.supported_extensions", {".jpg", ".png"})
        (tmp_path / "README.txt").touch()

        first_image = _get_first_image(tmp_path)

        assert first_image is None


class TestScanDatasets:
    def test_builds_id_to_path_mapping_and_pagination(self, tmp_path: Path, monkeypatch) -> None:
        dataset_dir = tmp_path / "aquarium"
        dataset_dir.mkdir()
        monkeypatch.setattr("domain.services.dataset_discovery.generate_image_thumbnail", lambda _path: "thumb")
        monkeypatch.setattr("domain.services.dataset_discovery.settings.supported_extensions", {".jpg", ".png"})
        (dataset_dir / "image.jpg").touch()

        datasets, dataset_paths = scan_datasets(tmp_path)

        assert len(datasets.datasets) == 1
        assert datasets.pagination.count == 1
        assert datasets.pagination.total == 1
        dataset_id = datasets.datasets[0].id
        assert dataset_id == uuid5(DATASET_NS, "aquarium")
        assert dataset_id in dataset_paths
        assert dataset_paths[dataset_id] == dataset_dir

    def test_sets_thumbnail_from_first_supported_image(self, tmp_path: Path, monkeypatch) -> None:
        dataset_dir = tmp_path / "aquarium"
        dataset_dir.mkdir()
        (dataset_dir / "README.txt").write_text("info")
        first_image = dataset_dir / "0001.jpg"
        first_image.touch()
        monkeypatch.setattr("domain.services.dataset_discovery.settings.supported_extensions", {".jpg", ".png"})
        monkeypatch.setattr(
            "domain.services.dataset_discovery.generate_image_thumbnail",
            lambda image_path: f"thumb::{image_path.name}",
        )

        datasets, _ = scan_datasets(tmp_path)

        assert len(datasets.datasets) == 1
        assert datasets.datasets[0].thumbnail == "thumb::0001.jpg"

    def test_sets_thumbnail_none_when_no_supported_images(self, tmp_path: Path, monkeypatch) -> None:
        dataset_dir = tmp_path / "aquarium"
        dataset_dir.mkdir()
        (dataset_dir / "README.txt").write_text("no images")
        monkeypatch.setattr("domain.services.dataset_discovery.settings.supported_extensions", {".jpg", ".png"})

        datasets, _ = scan_datasets(tmp_path)

        assert len(datasets.datasets) == 1
        assert datasets.datasets[0].thumbnail is None

    def test_raises_when_root_does_not_exist(self, tmp_path: Path) -> None:
        missing_dir = tmp_path / "missing"

        with pytest.raises(DatasetNotFoundError, match="does not exist"):
            scan_datasets(missing_dir)

    def test_raises_when_root_is_not_directory(self, tmp_path: Path) -> None:
        file_path = tmp_path / "datasets.txt"
        file_path.write_text("not a directory")

        with pytest.raises(DatasetNotFoundError, match="is not a directory"):
            scan_datasets(file_path)
