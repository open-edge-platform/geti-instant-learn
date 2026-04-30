# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from uuid import uuid4, uuid5

import pytest

from domain.errors import DatasetNotFoundError
from domain.services.dataset_discovery import (
    DATASET_NS,
    DatasetResolver,
    _get_first_image,
    scan_datasets,
)


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

    def test_default_cache_resolution_matches_sorted_scan_order(self, tmp_path: Path, monkeypatch) -> None:
        zebra_dir = tmp_path / "zebra"
        zebra_dir.mkdir()
        (zebra_dir / "image.jpg").touch()
        aquarium_dir = tmp_path / "aquarium"
        aquarium_dir.mkdir()
        (aquarium_dir / "image.jpg").touch()
        monkeypatch.setattr("domain.services.dataset_discovery.settings.supported_extensions", {".jpg", ".png"})
        monkeypatch.setattr("domain.services.dataset_discovery.generate_image_thumbnail", lambda _path: "thumb")

        datasets, _ = scan_datasets(tmp_path)
        resolver = DatasetResolver(tmp_path)

        assert [dataset.name for dataset in datasets.datasets] == ["Aquarium", "Zebra"]
        assert resolver.get_dataset_path(dataset_id=None) == aquarium_dir

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


class TestDatasetResolver:
    def test_initializes_cache_from_datasets_root(self, tmp_path: Path, monkeypatch) -> None:
        aquarium_dir = tmp_path / "aquarium"
        aquarium_dir.mkdir()
        (aquarium_dir / "image.jpg").touch()
        zebra_dir = tmp_path / "zebra"
        zebra_dir.mkdir()
        (zebra_dir / "image.jpg").touch()
        monkeypatch.setattr("domain.services.dataset_discovery.settings.supported_extensions", {".jpg", ".png"})
        monkeypatch.setattr("domain.services.dataset_discovery.generate_image_thumbnail", lambda _path: "thumb")

        resolver = DatasetResolver(tmp_path)

        # Verify that cache is built (we can test it by resolving paths)
        aquarium_id = uuid5(DATASET_NS, "aquarium")
        result = resolver.get_dataset_path(dataset_id=aquarium_id)
        assert result == aquarium_dir

    def test_get_dataset_path_returns_matching_dataset_for_id(self, tmp_path: Path, monkeypatch) -> None:
        dataset_dir = tmp_path / "aquarium"
        dataset_dir.mkdir()
        (dataset_dir / "image.jpg").touch()
        monkeypatch.setattr("domain.services.dataset_discovery.settings.supported_extensions", {".jpg", ".png"})
        monkeypatch.setattr("domain.services.dataset_discovery.generate_image_thumbnail", lambda _path: "thumb")

        resolver = DatasetResolver(tmp_path)
        dataset_id = uuid5(DATASET_NS, "aquarium")

        result = resolver.get_dataset_path(dataset_id=dataset_id)

        assert result == dataset_dir

    def test_get_dataset_path_returns_first_dataset_when_id_is_none(self, tmp_path: Path, monkeypatch) -> None:
        aquarium_dir = tmp_path / "aquarium"
        aquarium_dir.mkdir()
        (aquarium_dir / "image.jpg").touch()
        zebra_dir = tmp_path / "zebra"
        zebra_dir.mkdir()
        (zebra_dir / "image.jpg").touch()
        monkeypatch.setattr("domain.services.dataset_discovery.settings.supported_extensions", {".jpg", ".png"})
        monkeypatch.setattr("domain.services.dataset_discovery.generate_image_thumbnail", lambda _path: "thumb")

        resolver = DatasetResolver(tmp_path)

        result = resolver.get_dataset_path(dataset_id=None)

        assert result == aquarium_dir

    def test_get_dataset_path_raises_when_id_not_found(self, tmp_path: Path, monkeypatch) -> None:
        dataset_dir = tmp_path / "aquarium"
        dataset_dir.mkdir()
        (dataset_dir / "image.jpg").touch()
        monkeypatch.setattr("domain.services.dataset_discovery.settings.supported_extensions", {".jpg", ".png"})
        monkeypatch.setattr("domain.services.dataset_discovery.generate_image_thumbnail", lambda _path: "thumb")

        resolver = DatasetResolver(tmp_path)
        unknown_id = uuid4()

        with pytest.raises(DatasetNotFoundError, match="was not found"):
            resolver.get_dataset_path(dataset_id=unknown_id)

    def test_raises_when_datasets_root_does_not_exist(self, tmp_path: Path) -> None:
        missing_dir = tmp_path / "missing"

        with pytest.raises(DatasetNotFoundError, match="does not exist"):
            DatasetResolver(missing_dir)

    def test_raises_when_datasets_root_is_not_directory(self, tmp_path: Path) -> None:
        file_path = tmp_path / "datasets.txt"
        file_path.write_text("not a directory")

        with pytest.raises(DatasetNotFoundError, match="is not a directory"):
            DatasetResolver(file_path)
