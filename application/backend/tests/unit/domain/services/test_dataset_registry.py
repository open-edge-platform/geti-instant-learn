# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID, uuid4, uuid5

_DATASET_NS = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

import pytest

from domain.services.dataset_registry import DatasetRegistryService


def _make_service(datasets_root: Path) -> DatasetRegistryService:
    settings = SimpleNamespace(template_dataset_dir=datasets_root)
    with patch("domain.services.dataset_registry.get_settings", return_value=settings):
        return DatasetRegistryService()


def test_discovers_dataset_directories(tmp_path: Path):
    (tmp_path / "aquarium").mkdir()
    (tmp_path / "nuts").mkdir()
    result = _make_service(tmp_path).list_datasets()
    assert len(result.datasets) == 2
    assert {d.name for d in result.datasets} == {"Aquarium", "Nuts"}


def test_empty_when_root_does_not_exist(tmp_path: Path):
    result = _make_service(tmp_path / "nonexistent").list_datasets()
    assert result.datasets == []


def test_ignores_files_in_root(tmp_path: Path):
    (tmp_path / "aquarium").mkdir()
    (tmp_path / "readme.txt").write_text("ignore me")
    result = _make_service(tmp_path).list_datasets()
    assert len(result.datasets) == 1


def test_dataset_name_is_title_cased_from_directory(tmp_path: Path):
    (tmp_path / "coffee-beans").mkdir()
    result = _make_service(tmp_path).list_datasets()
    assert result.datasets[0].name == "Coffee Beans"


def test_description_is_generated_from_name(tmp_path: Path):
    (tmp_path / "aquarium").mkdir()
    result = _make_service(tmp_path).list_datasets()
    assert result.datasets[0].description == "This is sample dataset of aquarium."


def test_dataset_ids_are_stable_across_instances(tmp_path: Path):
    (tmp_path / "aquarium").mkdir()
    ids_a = {d.id for d in _make_service(tmp_path).list_datasets().datasets}
    ids_b = {d.id for d in _make_service(tmp_path).list_datasets().datasets}
    assert ids_a == ids_b


def test_dataset_id_derived_from_directory_name(tmp_path: Path):
    (tmp_path / "aquarium").mkdir()
    result = _make_service(tmp_path).list_datasets()
    assert result.datasets[0].id == uuid5(_DATASET_NS, "aquarium")


def test_get_dataset_path_returns_directory(tmp_path: Path):
    (tmp_path / "aquarium").mkdir()
    service = _make_service(tmp_path)
    dataset = service.list_datasets().datasets[0]
    assert service.get_dataset_path(dataset.id) == tmp_path / "aquarium"


def test_id_maps_to_correct_directory(tmp_path: Path):
    (tmp_path / "aquarium").mkdir()
    service = _make_service(tmp_path)
    expected_id = uuid5(_DATASET_NS, "aquarium")
    assert service.get_dataset_path(expected_id) == tmp_path / "aquarium"


def test_get_dataset_path_unknown_id_raises_key_error(tmp_path: Path):
    service = _make_service(tmp_path)
    with pytest.raises(KeyError):
        service.get_dataset_path(uuid4())
