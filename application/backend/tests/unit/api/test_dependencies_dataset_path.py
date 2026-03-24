# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException

from dependencies import get_dataset_path_by_id


def test_get_dataset_path_by_id_returns_path() -> None:
    dataset_id = uuid4()
    expected_path = Path("/tmp/datasets/aquarium")

    result = get_dataset_path_by_id(dataset_id=dataset_id, dataset_paths={dataset_id: expected_path})

    assert result == expected_path


def test_get_dataset_path_by_id_raises_404_for_unknown_id() -> None:
    unknown_id = uuid4()

    with pytest.raises(HTTPException) as exc_info:
        get_dataset_path_by_id(dataset_id=unknown_id, dataset_paths={})

    assert exc_info.value.status_code == 404
    assert str(unknown_id) in str(exc_info.value.detail)
