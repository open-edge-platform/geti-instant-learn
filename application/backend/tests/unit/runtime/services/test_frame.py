# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from runtime.services.frame import FrameService


@pytest.fixture
def frame_repository_mock():
    return MagicMock(name="FrameRepository")


@pytest.fixture
def frame_service(frame_repository_mock):
    return FrameService(frame_repo=frame_repository_mock)


def test_get_frame_path_returns_path(
    frame_service,
    frame_repository_mock,
):
    project_id = uuid.uuid4()
    frame_id = uuid.uuid4()
    expected_path = Path(f"/fake/path/{project_id}/frames/{frame_id}.jpg")
    frame_repository_mock.get_frame_path.return_value = expected_path

    result_path = frame_service.get_frame_path(project_id, frame_id)

    assert result_path == expected_path
    frame_repository_mock.get_frame_path.assert_called_once_with(project_id, frame_id)


def test_get_frame_path_returns_none_when_not_exists(
    frame_service,
    frame_repository_mock,
):
    project_id = uuid.uuid4()
    frame_id = uuid.uuid4()
    frame_repository_mock.get_frame_path.return_value = None

    result_path = frame_service.get_frame_path(project_id, frame_id)

    assert result_path is None
    frame_repository_mock.get_frame_path.assert_called_once_with(project_id, frame_id)
