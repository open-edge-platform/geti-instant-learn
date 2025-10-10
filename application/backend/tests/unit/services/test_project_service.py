# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.errors import ResourceAlreadyExistsError, ResourceNotFoundError
from services.project import ProjectService
from services.schemas.project import ProjectCreateSchema, ProjectSchema, ProjectUpdateSchema


def make_project(
    *,
    project_id=None,
    name="proj",
    active=False,
):
    if project_id is None:
        project_id = uuid.uuid4()
    return SimpleNamespace(id=project_id, name=name, active=active)


@pytest.fixture
def session_mock():
    return MagicMock(name="session")


@pytest.fixture
def repo_mock():
    return MagicMock(name="ProjectRepositoryMock")


@pytest.fixture
def service(session_mock, repo_mock):
    svc = ProjectService(session=session_mock)
    svc.project_repository = repo_mock
    return svc


@pytest.mark.parametrize("explicit_id", [None, uuid.uuid4()])
def test_create_project_success(service, repo_mock, session_mock, explicit_id):
    if explicit_id is None:
        data = ProjectCreateSchema(name="alpha")
    else:
        data = ProjectCreateSchema(id=explicit_id, name="alpha")
    repo_mock.exists_by_name.return_value = False
    repo_mock.exists_by_id.return_value = False
    repo_mock.get_active.return_value = None

    result = service.create_project(data)

    assert isinstance(result, ProjectSchema)
    assert result.name == "alpha"
    repo_mock.add.assert_called_once()
    assert session_mock.flush.call_count == 2  # flushes: initial add + activation
    session_mock.commit.assert_called_once()
    session_mock.refresh.assert_called_once()
    repo_mock.exists_by_name.assert_called_once_with("alpha")
    repo_mock.exists_by_id.assert_called_once_with(data.id)


def test_create_project_duplicate_name(service, repo_mock, session_mock):
    data = ProjectCreateSchema(name="dup")
    repo_mock.exists_by_name.return_value = True

    with pytest.raises(ResourceAlreadyExistsError):
        service.create_project(data)

    repo_mock.add.assert_not_called()
    session_mock.commit.assert_not_called()


def test_create_project_duplicate_id(service, repo_mock, session_mock):
    pid = uuid.uuid4()
    data = ProjectCreateSchema(id=pid, name="x")
    repo_mock.exists_by_name.return_value = False
    repo_mock.exists_by_id.return_value = True

    with pytest.raises(ResourceAlreadyExistsError):
        service.create_project(data)

    repo_mock.add.assert_not_called()
    session_mock.commit.assert_not_called()


def test_get_project_success(service, repo_mock):
    pid = uuid.uuid4()
    project = make_project(project_id=pid)
    repo_mock.get_by_id.return_value = project

    result = service.get_project(pid)
    assert isinstance(result, ProjectSchema)
    assert result.id == pid
    repo_mock.get_by_id.assert_called_once_with(pid)


def test_get_project_not_found(service, repo_mock):
    pid = uuid.uuid4()
    repo_mock.get_by_id.return_value = None
    with pytest.raises(ResourceNotFoundError):
        service.get_project(pid)


def test_list_projects(service, repo_mock):
    p1, p2 = make_project(), make_project()
    repo_mock.get_all.return_value = [p1, p2]

    result = service.list_projects()
    assert len(result.projects) == 2
    ids = {p.id for p in result.projects}
    assert ids == {p1.id, p2.id}
    repo_mock.get_all.assert_called_once()


def test_update_project_success(service, repo_mock, session_mock):
    pid = uuid.uuid4()
    existing = make_project(project_id=pid, name="old")
    repo_mock.get_by_id.return_value = existing
    repo_mock.exists_by_name.return_value = False

    data = ProjectUpdateSchema(name="new", active=existing.active)
    updated = service.update_project(pid, data)

    assert updated.name == "new"
    session_mock.commit.assert_called_once()
    session_mock.refresh.assert_called_once()


def test_update_project_duplicate_name(service, repo_mock, session_mock):
    pid = uuid.uuid4()
    existing = make_project(project_id=pid, name="old")
    repo_mock.get_by_id.return_value = existing
    repo_mock.exists_by_name.return_value = True

    with pytest.raises(ResourceAlreadyExistsError):
        service.update_project(pid, ProjectUpdateSchema(name="other", active=existing.active))

    session_mock.commit.assert_not_called()


def test_update_project_not_found(service, repo_mock):
    repo_mock.get_by_id.return_value = None
    with pytest.raises(ResourceNotFoundError):
        service.update_project(uuid.uuid4(), ProjectUpdateSchema(name="x", active=False))


def test_set_active_project_success(service, repo_mock, session_mock):
    target = make_project(project_id=uuid.uuid4(), active=False)
    previously_active = make_project(project_id=uuid.uuid4(), active=True)
    repo_mock.get_by_id.return_value = target
    repo_mock.get_active.return_value = previously_active

    service.set_active_project(target.id)

    assert target.active is True
    assert previously_active.active is False
    session_mock.commit.assert_called_once()
    session_mock.refresh.assert_not_called()


def test_set_active_project_not_found(service, repo_mock):
    repo_mock.get_by_id.return_value = None
    with pytest.raises(ResourceNotFoundError):
        service.set_active_project(uuid.uuid4())


def test_get_active_project_success(service, repo_mock):
    active = make_project(active=True)
    repo_mock.get_active.return_value = active
    result = service.get_active_project_info()
    assert result.id == active.id
    assert result.name == active.name


def test_get_active_project_not_found(service, repo_mock):
    repo_mock.get_active.return_value = None
    with pytest.raises(ResourceNotFoundError):
        service.get_active_project_info()


def test_delete_project_success(service, repo_mock, session_mock):
    pid = uuid.uuid4()
    project = make_project(project_id=pid)
    repo_mock.get_by_id.return_value = project

    service.delete_project(pid)

    repo_mock.get_by_id.assert_called_once_with(pid)
    repo_mock.delete.assert_called_once_with(project)
    session_mock.commit.assert_called_once()


def test_delete_project_not_found(service, repo_mock):
    repo_mock.get_by_id.return_value = None
    with pytest.raises(ResourceNotFoundError):
        service.delete_project(uuid.uuid4())
