# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import IntegrityError

from domain.dispatcher import ComponentConfigChangeEvent
from domain.errors import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
    ResourceUpdateConflictError,
)
from domain.services.schemas.reader import SourceType, VideoFileConfig, WebCamConfig
from domain.services.schemas.source import SourceCreateSchema, SourceUpdateSchema
from domain.services.source import SourceService


def make_project(project_id=None, name="proj"):
    return SimpleNamespace(id=project_id or uuid.uuid4(), name=name)


def make_source(
    *,
    source_id=None,
    project_id=None,
    source_type: SourceType = SourceType.WEBCAM,
    config_extra: dict | None = None,
    connected: bool = False,
):
    base_cfg = {"source_type": source_type.value}
    if source_type == SourceType.WEBCAM:
        base_cfg |= {"device_id": 0}
    elif source_type == SourceType.VIDEO_FILE:
        base_cfg |= {"video_path": "/tmp/video.mp4"}
    if config_extra:
        base_cfg |= config_extra
    return SimpleNamespace(
        id=source_id or uuid.uuid4(),
        project_id=project_id or uuid.uuid4(),
        config=base_cfg,
        connected=connected,
    )


@pytest.fixture
def dispatcher_mock():
    return MagicMock(name="ConfigChangeDispatcher")


@pytest.fixture
def service(dispatcher_mock):
    session = MagicMock(name="SessionMock")
    project_repo = MagicMock(name="ProjectRepositoryMock")
    source_repo = MagicMock(name="SourceRepositoryMock")
    return SourceService(
        session=session,
        project_repository=project_repo,
        source_repository=source_repo,
        config_change_dispatcher=dispatcher_mock,
    )


def test_list_sources_success(service):
    project_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    s1 = make_source(project_id=project_id)
    s2 = make_source(project_id=project_id, source_type=SourceType.VIDEO_FILE)
    service.source_repository.get_all_by_project.return_value = [s1, s2]

    with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):
        result = service.list_sources(project_id)

    assert len(result.sources) == 2
    service.project_repository.get_by_id.assert_called_once_with(project_id)
    service.source_repository.get_all_by_project.assert_called_once_with(project_id)


def test_list_sources_empty_list(service):
    project_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    service.source_repository.get_all_by_project.return_value = []

    result = service.list_sources(project_id)

    assert result.sources == []
    service.project_repository.get_by_id.assert_called_once_with(project_id)
    service.source_repository.get_all_by_project.assert_called_once_with(project_id)


def test_get_source_success(service):
    project_id = uuid.uuid4()
    source = make_source(project_id=project_id)
    service.project_repository.get_by_id.return_value = make_project(project_id)
    service.source_repository.get_by_id_and_project.return_value = source

    schema = service.get_source(project_id=project_id, source_id=source.id)

    assert schema.id == source.id
    assert schema.config.source_type == SourceType(source.config["source_type"])
    service.source_repository.get_by_id_and_project.assert_called_once_with(source_id=source.id, project_id=project_id)


def test_get_source_not_found(service):
    project_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    service.source_repository.get_by_id_and_project.return_value = None

    with pytest.raises(ResourceNotFoundError):
        service.get_source(project_id=project_id, source_id=uuid.uuid4())


def test_create_source_success(service, dispatcher_mock):
    new_id = uuid.uuid4()
    project_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    service.source_repository.get_connected_in_project.return_value = None
    create_schema = SourceCreateSchema(
        id=new_id,
        connected=True,
        config=WebCamConfig(source_type=SourceType.WEBCAM, name="Webcam A", device_id=2),
    )

    result = service.create_source(project_id=project_id, create_data=create_schema)

    assert result.id == new_id
    assert result.connected is True
    assert result.config.device_id == 2
    service.source_repository.add.assert_called_once()
    service.session.commit.assert_called_once()
    service.session.refresh.assert_called_once()
    dispatcher_mock.dispatch.assert_called_once()
    ev = dispatcher_mock.dispatch.call_args_list[0].args[0]
    assert isinstance(ev, ComponentConfigChangeEvent)
    assert ev.project_id == project_id
    assert ev.component_type == "source"
    assert ev.component_id == str(new_id)


def test_create_source_type_conflict_raises_integrity_error(service):
    project_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    service.source_repository.get_connected_in_project.return_value = None

    create_schema = SourceCreateSchema(
        id=uuid.uuid4(),
        connected=False,
        config=WebCamConfig(source_type=SourceType.WEBCAM, name="Dup", device_id=0),
    )

    mock_error = IntegrityError("statement", "params", "orig")
    mock_error.orig = Exception("UNIQUE constraint failed: uq_source_type_per_project")
    service.session.commit.side_effect = mock_error

    with pytest.raises(ResourceAlreadyExistsError) as exc_info:
        service.create_source(project_id=project_id, create_data=create_schema)

    assert exc_info.value.resource_type == ResourceType.SOURCE
    assert exc_info.value.field == "source_type"
    assert "source of type 'webcam' already exists" in str(exc_info.value).lower()
    service.session.rollback.assert_called_once()


def test_create_source_name_conflict_raises_integrity_error(service):
    project_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    service.source_repository.get_connected_in_project.return_value = None

    create_schema = SourceCreateSchema(
        id=uuid.uuid4(),
        connected=False,
        config=WebCamConfig(source_type=SourceType.WEBCAM, name="DupName", device_id=0),
    )

    mock_error = IntegrityError("statement", "params", "orig")
    mock_error.orig = Exception("UNIQUE constraint failed: uq_source_name_per_project")
    service.session.commit.side_effect = mock_error

    with pytest.raises(ResourceAlreadyExistsError) as exc_info:
        service.create_source(project_id=project_id, create_data=create_schema)

    assert exc_info.value.resource_type == ResourceType.SOURCE
    assert exc_info.value.field == "name"
    assert "source with" in str(exc_info.value).lower()
    assert "this name" in str(exc_info.value).lower()
    assert "already exists in this project" in str(exc_info.value).lower()
    service.session.rollback.assert_called_once()


def test_create_source_disconnects_previous_connected(service):
    project_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    prev_connected = make_source(project_id=project_id, connected=True)
    service.source_repository.get_connected_in_project.return_value = prev_connected
    create_schema = SourceCreateSchema(
        id=uuid.uuid4(),
        connected=True,
        config=WebCamConfig(source_type=SourceType.WEBCAM, name="Primary", device_id=1),
    )

    service.create_source(project_id=project_id, create_data=create_schema)

    assert prev_connected.connected is False


def test_create_connected_source_violates_single_connected_constraint(service, tmp_path):
    project_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)

    service.source_repository.get_connected_in_project.return_value = None  # assume no connected source found

    # Create a temporary video file for validation
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"fake video content")

    create_schema = SourceCreateSchema(
        id=uuid.uuid4(),
        connected=True,
        config=VideoFileConfig(source_type=SourceType.VIDEO_FILE, video_path=str(video_file)),
    )

    # simulate IntegrityError from database constraint
    mock_error = IntegrityError("statement", "params", "orig")
    mock_error.orig = Exception("UNIQUE constraint failed: uq_single_connected_source_per_project")
    service.session.commit.side_effect = mock_error

    with pytest.raises(ResourceAlreadyExistsError) as exc_info:
        service.create_source(project_id=project_id, create_data=create_schema)

    assert exc_info.value.resource_type == ResourceType.SOURCE
    assert exc_info.value.field == "connected"
    assert "only one source can be connected per project" in str(exc_info.value).lower()
    service.session.rollback.assert_called_once()


def test_update_source_success(service, dispatcher_mock):
    project_id = uuid.uuid4()
    source_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    existing = make_source(project_id=project_id, source_id=source_id, source_type=SourceType.WEBCAM, connected=False)
    service.source_repository.get_by_id_and_project.return_value = existing
    prev_connected = make_source(project_id=project_id, connected=True)
    service.source_repository.get_connected_in_project.return_value = prev_connected
    update_schema = SourceUpdateSchema(
        connected=True,
        config=WebCamConfig(source_type=SourceType.WEBCAM, name="Renamed", device_id=5),
    )

    result = service.update_source(project_id=project_id, source_id=source_id, update_data=update_schema)

    assert result.id == source_id
    assert existing.connected is True
    assert existing.config["device_id"] == 5
    assert prev_connected.connected is False
    service.session.commit.assert_called_once()
    service.session.refresh.assert_called_once_with(existing)
    dispatcher_mock.dispatch.assert_called_once()
    ev = dispatcher_mock.dispatch.call_args_list[0].args[0]
    assert isinstance(ev, ComponentConfigChangeEvent)
    assert ev.project_id == project_id
    assert ev.component_type == "source"
    assert ev.component_id == str(source_id)


def test_update_source_type_change_conflict(service, tmp_path):
    project_id = uuid.uuid4()
    source_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    existing = make_source(project_id=project_id, source_id=source_id, source_type=SourceType.WEBCAM)
    service.source_repository.get_by_id_and_project.return_value = existing

    # Create a temporary video file for validation
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"fake video content")

    update_schema = SourceUpdateSchema(
        connected=False,
        config=VideoFileConfig(source_type=SourceType.VIDEO_FILE, video_path=str(video_file)),
    )

    with pytest.raises(ResourceUpdateConflictError):
        service.update_source(project_id=project_id, source_id=source_id, update_data=update_schema)

    service.session.commit.assert_not_called()


def test_update_source_not_found(service):
    project_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    service.source_repository.get_by_id_and_project.return_value = None
    update_schema = SourceUpdateSchema(
        connected=False,
        config=WebCamConfig(source_type=SourceType.WEBCAM, name="X", device_id=0),
    )

    with pytest.raises(ResourceNotFoundError):
        service.update_source(project_id=project_id, source_id=uuid.uuid4(), update_data=update_schema)


def test_delete_source_success(service, dispatcher_mock):
    project_id = uuid.uuid4()
    source_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    existing = make_source(project_id=project_id, source_id=source_id)
    service.source_repository.get_by_id_and_project.return_value = existing

    service.delete_source(project_id=project_id, source_id=source_id)

    service.source_repository.delete.assert_called_once_with(existing)
    service.session.commit.assert_called_once()
    dispatcher_mock.dispatch.assert_called_once()
    ev = dispatcher_mock.dispatch.call_args_list[0].args[0]
    assert isinstance(ev, ComponentConfigChangeEvent)
    assert ev.project_id == project_id
    assert ev.component_type == "source"
    assert ev.component_id == str(source_id)


def test_delete_source_not_found(service, dispatcher_mock):
    project_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    service.source_repository.get_by_id_and_project.return_value = None

    with pytest.raises(ResourceNotFoundError):
        service.delete_source(project_id=project_id, source_id=uuid.uuid4())

    dispatcher_mock.dispatch.assert_not_called()


def test_delete_source_project_not_found(service, dispatcher_mock):
    service.project_repository.get_by_id.return_value = None
    with pytest.raises(ResourceNotFoundError):
        service.delete_source(uuid.uuid4(), uuid.uuid4())
    dispatcher_mock.dispatch.assert_not_called()


def test_create_source_emits_event_when_connected_false(service, dispatcher_mock):
    project_id = uuid.uuid4()
    new_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    service.source_repository.get_connected_in_project.return_value = None
    create_schema = SourceCreateSchema(
        id=new_id,
        connected=False,
        config=WebCamConfig(source_type=SourceType.WEBCAM, device_id=3),
    )

    service.create_source(project_id=project_id, create_data=create_schema)

    dispatcher_mock.dispatch.assert_called_once()
    ev = dispatcher_mock.dispatch.call_args_list[0].args[0]
    assert isinstance(ev, ComponentConfigChangeEvent)
    assert ev.project_id == project_id
    assert ev.component_type == "source"
    assert ev.component_id == str(new_id)


def test_update_source_emits_event_when_no_connection_change(service, dispatcher_mock):
    project_id = uuid.uuid4()
    source_id = uuid.uuid4()
    service.project_repository.get_by_id.return_value = make_project(project_id)
    existing = make_source(project_id=project_id, source_id=source_id, source_type=SourceType.WEBCAM, connected=True)
    service.source_repository.get_by_id_and_project.return_value = existing
    service.source_repository.get_connected_in_project.return_value = existing  # already connected
    update_schema = SourceUpdateSchema(
        connected=True,
        config=WebCamConfig(source_type=SourceType.WEBCAM, device_id=7),
    )

    service.update_source(project_id=project_id, source_id=source_id, update_data=update_schema)

    dispatcher_mock.dispatch.assert_called_once()
    ev = dispatcher_mock.dispatch.call_args_list[0].args[0]
    assert isinstance(ev, ComponentConfigChangeEvent)
    assert ev.project_id == project_id
    assert ev.component_id == str(source_id)
