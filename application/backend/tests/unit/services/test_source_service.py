# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.components.schemas.reader import SourceType
from services.common import (
    ResourceNotFoundError,
    ResourceUpdateConflictError,
)
from services.schemas.source import WebcamSourcePayload
from services.source import SourceService


def make_project(project_id=None):
    return SimpleNamespace(id=project_id or uuid.uuid4(), name="proj")


def make_source(
    *,
    source_id=None,
    project_id=None,
    name="cam",
    source_type=SourceType.WEBCAM,
    config=None,
    connected=False,
):
    return SimpleNamespace(
        id=source_id or uuid.uuid4(),
        project_id=project_id or uuid.uuid4(),
        name=name,
        type=source_type,
        config=config or {"device_id": 0},
        connected=connected,
    )


@pytest.fixture
def session_mock():
    return MagicMock(name="session")


@pytest.fixture
def source_repo_mock():
    return MagicMock(name="SourceRepositoryMock")


@pytest.fixture
def project_repo_mock():
    return MagicMock(name="ProjectRepositoryMock")


@pytest.fixture
def service(session_mock, source_repo_mock, project_repo_mock):
    svc = SourceService(session=session_mock)
    svc.source_repository = source_repo_mock
    svc.project_repository = project_repo_mock
    return svc


def test_list_sources_success(service, project_repo_mock, source_repo_mock):
    pid = uuid.uuid4()
    project_repo_mock.get_by_id.return_value = make_project(pid)
    s1 = make_source(project_id=pid)
    s2 = make_source(project_id=pid)
    source_repo_mock.get_all_by_project.return_value = [s1, s2]

    result = service.list_sources(pid)

    assert len(result.sources) == 2
    project_repo_mock.get_by_id.assert_called_once_with(pid)
    source_repo_mock.get_all_by_project.assert_called_once_with(pid)


def test_list_sources_project_not_found(service, project_repo_mock):
    project_repo_mock.get_by_id.return_value = None
    with pytest.raises(ResourceNotFoundError):
        service.list_sources(uuid.uuid4())


def test_get_by_id_found(service, source_repo_mock):
    src = make_source()
    source_repo_mock.get_by_id.return_value = src

    schema = service.get_by_id(src.id)

    assert schema is not None
    assert schema.id == src.id
    assert schema.name == src.name
    source_repo_mock.get_by_id.assert_called_once_with(src.id)


def test_get_by_id_not_found(service, source_repo_mock):
    source_repo_mock.get_by_id.return_value = None
    assert service.get_by_id(uuid.uuid4()) is None


def test_get_by_id_and_project_found(service, source_repo_mock):
    pid = uuid.uuid4()
    src = make_source(project_id=pid)
    source_repo_mock.get_by_id_and_project.return_value = src

    schema = service.get_by_id_and_project(src.id, pid)

    assert schema is not None
    assert schema.id == src.id
    source_repo_mock.get_by_id_and_project.assert_called_once_with(source_id=src.id, project_id=pid)


def test_get_by_id_and_project_not_found(service, source_repo_mock):
    source_repo_mock.get_by_id_and_project.return_value = None
    assert service.get_by_id_and_project(uuid.uuid4(), uuid.uuid4()) is None


def test_upsert_create_success(service, project_repo_mock, source_repo_mock, session_mock):
    pid = uuid.uuid4()
    sid = uuid.uuid4()
    project_repo_mock.get_by_id.return_value = make_project(pid)
    source_repo_mock.get_by_id_and_project.return_value = None
    payload = WebcamSourcePayload(source_type=SourceType.WEBCAM, name="webcam0", device_id=3)

    schema, created = service.upsert_source(project_id=pid, source_id=sid, payload=payload)

    assert created is True
    assert schema.id == sid
    assert schema.name == "webcam0"
    assert schema.device_id == 3
    source_repo_mock.add.assert_called_once()
    session_mock.commit.assert_called_once()
    session_mock.refresh.assert_called_once()


def test_upsert_update_success(service, project_repo_mock, source_repo_mock, session_mock):
    pid = uuid.uuid4()
    sid = uuid.uuid4()
    project_repo_mock.get_by_id.return_value = make_project(pid)
    existing = make_source(source_id=sid, project_id=pid, name="old", config={"device_id": 1})
    source_repo_mock.get_by_id_and_project.return_value = existing
    payload = WebcamSourcePayload(source_type=SourceType.WEBCAM, name="new", device_id=7)

    schema, created = service.upsert_source(project_id=pid, source_id=sid, payload=payload)

    assert created is False
    assert existing.name == "new"
    assert existing.config == {"device_id": 7}
    assert schema.device_id == 7
    source_repo_mock.add.assert_not_called()
    session_mock.commit.assert_called_once()
    session_mock.refresh.assert_called_once_with(existing)


def test_upsert_update_type_conflict(service, project_repo_mock, source_repo_mock, session_mock):
    pid = uuid.uuid4()
    sid = uuid.uuid4()
    project_repo_mock.get_by_id.return_value = make_project(pid)
    existing = make_source(source_id=sid, project_id=pid, source_type=SourceType.WEBCAM)
    source_repo_mock.get_by_id_and_project.return_value = existing
    # fabricate a payload with a different type (simulate future type)
    payload = SimpleNamespace(
        source_type="ip_camera",  # different
        name="cam",
        device_id=0,
        model_dump=lambda: {"source_type": "ip_camera", "name": "cam", "device_id": 0},
    )

    with pytest.raises(ResourceUpdateConflictError):
        service.upsert_source(project_id=pid, source_id=sid, payload=payload)

    session_mock.commit.assert_not_called()


def test_upsert_project_not_found(service, project_repo_mock):
    project_repo_mock.get_by_id.return_value = None
    with pytest.raises(ResourceNotFoundError):
        service.upsert_source(
            project_id=uuid.uuid4(),
            source_id=uuid.uuid4(),
            payload=WebcamSourcePayload(source_type=SourceType.WEBCAM, name="x", device_id=0),
        )


def test_delete_source_success(service, project_repo_mock, source_repo_mock, session_mock):
    pid = uuid.uuid4()
    sid = uuid.uuid4()
    project_repo_mock.get_by_id.return_value = make_project(pid)
    src = make_source(source_id=sid, project_id=pid)
    source_repo_mock.get_by_id_and_project.return_value = src

    service.delete_source(project_id=pid, source_id=sid)

    source_repo_mock.delete.assert_called_once_with(src)
    session_mock.commit.assert_called_once()


def test_delete_source_project_not_found(service, project_repo_mock):
    project_repo_mock.get_by_id.return_value = None
    with pytest.raises(ResourceNotFoundError):
        service.delete_source(project_id=uuid.uuid4(), source_id=uuid.uuid4())


def test_delete_source_not_found(service, project_repo_mock, source_repo_mock):
    pid = uuid.uuid4()
    project_repo_mock.get_by_id.return_value = make_project(pid)
    source_repo_mock.get_by_id_and_project.return_value = None
    with pytest.raises(ResourceNotFoundError):
        service.delete_source(project_id=pid, source_id=uuid.uuid4())
