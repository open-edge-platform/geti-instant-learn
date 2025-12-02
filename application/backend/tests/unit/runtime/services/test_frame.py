# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import uuid
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from domain.errors import ResourceNotFoundError, ResourceType, ServiceError
from domain.services.schemas.processor import InputData
from runtime.errors import PipelineNotActiveError
from runtime.services.frame import FrameService


@pytest.fixture
def sample_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = [255, 0, 0]  # Blue in BGR
    return frame


@pytest.fixture
def sample_input_data(sample_frame):
    return InputData(
        timestamp=1234567890000,
        frame=sample_frame,
        context={"source": "test_camera"},
    )


@pytest.fixture
def frame_repository_mock():
    return MagicMock(name="FrameRepository")


@pytest.fixture
def project_repository_mock():
    return MagicMock(name="ProjectRepository")


@pytest.fixture
def source_repository_mock():
    return MagicMock(name="SourceRepository")


@pytest.fixture
def frame_service_without_queue(
    frame_repository_mock,
    project_repository_mock,
    source_repository_mock,
):
    """Frame service without queue (for GET requests)."""
    return FrameService(
        frame_repo=frame_repository_mock,
        project_repo=project_repository_mock,
        source_repo=source_repository_mock,
    )


@pytest.fixture
def frame_service_with_queue(
    frame_repository_mock,
    project_repository_mock,
    source_repository_mock,
    sample_input_data,
):
    """Frame service with queue (for POST requests)."""
    consumer_queue = Queue()
    consumer_queue.put(sample_input_data)
    return FrameService(
        frame_repo=frame_repository_mock,
        project_repo=project_repository_mock,
        source_repo=source_repository_mock,
        inbound_queue=consumer_queue,
    )


def make_project(project_id=None, name="test_project", active=True):
    if project_id is None:
        project_id = uuid.uuid4()
    return SimpleNamespace(id=project_id, name=name, active=active)


def make_source(source_id=None, connected=True):
    if source_id is None:
        source_id = uuid.uuid4()
    return SimpleNamespace(
        id=source_id,
        connected=connected,
        config={"source_type": "webcam", "device_id": 0},
    )


def test_capture_frame_without_queue_raises_error(
    frame_service_without_queue,
):
    project_id = uuid.uuid4()

    with pytest.raises(ServiceError, match="not been properly initialized with inbound queue"):
        frame_service_without_queue.capture_frame(project_id)


def test_capture_frame_success(
    frame_service_with_queue,
    frame_repository_mock,
    project_repository_mock,
    source_repository_mock,
    sample_input_data,
):
    project_id = uuid.uuid4()
    expected_project = make_project(project_id=project_id, active=True)
    expected_source = make_source(connected=True)

    project_repository_mock.get_by_id.return_value = expected_project
    source_repository_mock.get_active_in_project.return_value = expected_source

    captured_frame_id = frame_service_with_queue.capture_frame(project_id)

    assert isinstance(captured_frame_id, uuid.UUID)
    project_repository_mock.get_by_id.assert_called_once_with(project_id)
    source_repository_mock.get_active_in_project.assert_called_once_with(project_id)
    frame_repository_mock.save_frame.assert_called_once()

    call_args = frame_repository_mock.save_frame.call_args
    assert call_args[0][0] == project_id
    assert isinstance(call_args[0][1], uuid.UUID)
    np.testing.assert_array_equal(call_args[0][2], sample_input_data.frame)


def test_capture_frame_project_not_found(
    frame_service_with_queue,
    project_repository_mock,
):
    project_id = uuid.uuid4()
    project_repository_mock.get_by_id.return_value = None

    with pytest.raises(ResourceNotFoundError) as exc_info:
        frame_service_with_queue.capture_frame(project_id)

    assert exc_info.value.resource_type == ResourceType.PROJECT
    assert exc_info.value.resource_id == str(project_id)


def test_capture_frame_project_not_active(
    frame_service_with_queue,
    project_repository_mock,
):
    project_id = uuid.uuid4()
    inactive_project = make_project(project_id=project_id, active=False)
    project_repository_mock.get_by_id.return_value = inactive_project

    with pytest.raises(PipelineNotActiveError, match="project .* is not active"):
        frame_service_with_queue.capture_frame(project_id)


def test_capture_frame_no_connected_source(
    frame_service_with_queue,
    project_repository_mock,
    source_repository_mock,
):
    project_id = uuid.uuid4()
    active_project = make_project(project_id=project_id, active=True)
    project_repository_mock.get_by_id.return_value = active_project
    source_repository_mock.get_active_in_project.return_value = None

    with pytest.raises(ResourceNotFoundError) as exc_info:
        frame_service_with_queue.capture_frame(project_id)

    assert exc_info.value.resource_type == ResourceType.SOURCE
    assert "no connected source" in str(exc_info.value).lower()


def test_capture_frame_timeout(
    project_repository_mock,
    source_repository_mock,
    frame_repository_mock,
):
    project_id = uuid.uuid4()
    active_project = make_project(project_id=project_id, active=True)
    connected_source = make_source(connected=True)
    empty_queue = Queue()

    project_repository_mock.get_by_id.return_value = active_project
    source_repository_mock.get_active_in_project.return_value = connected_source

    frame_service = FrameService(
        frame_repo=frame_repository_mock,
        project_repo=project_repository_mock,
        source_repo=source_repository_mock,
        inbound_queue=empty_queue,
    )

    with pytest.raises(ServiceError, match="No frame received within 5 seconds"):
        frame_service.capture_frame(project_id)


def test_capture_frame_save_failure(
    frame_repository_mock,
    project_repository_mock,
    source_repository_mock,
    sample_input_data,
):
    project_id = uuid.uuid4()
    active_project = make_project(project_id=project_id, active=True)
    connected_source = make_source(connected=True)
    consumer_queue = Queue()
    consumer_queue.put(sample_input_data)

    project_repository_mock.get_by_id.return_value = active_project
    source_repository_mock.get_active_in_project.return_value = connected_source
    frame_repository_mock.save_frame.side_effect = RuntimeError("Disk full")

    frame_service = FrameService(
        frame_repo=frame_repository_mock,
        project_repo=project_repository_mock,
        source_repo=source_repository_mock,
        inbound_queue=consumer_queue,
    )

    with pytest.raises(ServiceError, match="Frame capture failed"):
        frame_service.capture_frame(project_id)


def test_capture_frame_queue_get_exception(
    project_repository_mock,
    source_repository_mock,
    frame_repository_mock,
):
    project_id = uuid.uuid4()
    active_project = make_project(project_id=project_id, active=True)
    connected_source = make_source(connected=True)
    consumer_queue = MagicMock(spec=Queue)
    consumer_queue.get.side_effect = RuntimeError("Queue error")

    project_repository_mock.get_by_id.return_value = active_project
    source_repository_mock.get_active_in_project.return_value = connected_source

    frame_service = FrameService(
        frame_repo=frame_repository_mock,
        project_repo=project_repository_mock,
        source_repo=source_repository_mock,
        inbound_queue=consumer_queue,
    )

    with pytest.raises(ServiceError, match="Frame capture failed"):
        frame_service.capture_frame(project_id)


def test_get_frame_path_returns_path(
    frame_service_without_queue,
    frame_repository_mock,
):
    project_id = uuid.uuid4()
    frame_id = uuid.uuid4()
    expected_path = Path(f"/fake/path/{project_id}/frames/{frame_id}.jpg")
    frame_repository_mock.get_frame_path.return_value = expected_path

    result_path = frame_service_without_queue.get_frame_path(project_id, frame_id)

    assert result_path == expected_path
    frame_repository_mock.get_frame_path.assert_called_once_with(project_id, frame_id)


def test_get_frame_path_returns_none_when_not_exists(
    frame_service_without_queue,
    frame_repository_mock,
):
    project_id = uuid.uuid4()
    frame_id = uuid.uuid4()
    frame_repository_mock.get_frame_path.return_value = None

    result_path = frame_service_without_queue.get_frame_path(project_id, frame_id)

    assert result_path is None
    frame_repository_mock.get_frame_path.assert_called_once_with(project_id, frame_id)
