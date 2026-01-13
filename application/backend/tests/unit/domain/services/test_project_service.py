# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from domain.dispatcher import ProjectActivationEvent, ProjectDeactivationEvent
from domain.errors import ResourceAlreadyExistsError, ResourceNotFoundError
from domain.services.project import ProjectService
from domain.services.schemas.pipeline import PipelineConfig
from domain.services.schemas.project import ProjectCreateSchema, ProjectSchema, ProjectUpdateSchema


def make_active_source(device_id: int = 0):
    return SimpleNamespace(
        id=uuid.uuid4(),
        active=True,
        config={"source_type": "usb_camera", "device_id": device_id},
    )


def make_sink(active: bool = True):
    return SimpleNamespace(
        id=uuid.uuid4(),
        active=active,
        config={"sink_type": "mqtt"},
    )


def make_model(project_id: uuid.UUID, active: bool = True):
    return SimpleNamespace(
        id=uuid.uuid4(),
        project_id=project_id,
        active=active,
        config={"model_type": "matcher"},
    )


def make_project(
    *,
    project_id=None,
    name="proj",
    active=False,
    sources=None,
    processors=None,
    sinks=None,
):
    if project_id is None:
        project_id = uuid.uuid4()
    if sources is None:
        sources = []
    if processors is None:
        processors = []
    if sinks is None:
        sinks = []
    return SimpleNamespace(id=project_id, name=name, active=active, sources=sources, processors=processors, sinks=sinks)


@pytest.fixture
def session_mock():
    return MagicMock(name="session")


@pytest.fixture
def repo_mock():
    return MagicMock(name="ProjectRepositoryMock")


@pytest.fixture
def dispatcher_mock():
    return MagicMock(name="ConfigChangeDispatcher")


@pytest.fixture
def service(session_mock, repo_mock, dispatcher_mock):
    svc = ProjectService(session=session_mock, config_change_dispatcher=dispatcher_mock)
    svc.project_repository = repo_mock
    return svc


@pytest.mark.parametrize("explicit_id", [None, uuid.uuid4()])
def test_create_project_success(service, repo_mock, session_mock, explicit_id):
    if explicit_id is None:
        data = ProjectCreateSchema(name="alpha")
    else:
        data = ProjectCreateSchema(id=explicit_id, name="alpha")
    repo_mock.get_active.return_value = None

    result = service.create_project(data)

    assert isinstance(result, ProjectSchema)
    assert result.name == "alpha"
    assert result.active is True
    repo_mock.add.assert_called_once()
    session_mock.commit.assert_called_once()
    session_mock.refresh.assert_called_once()


def test_create_project_duplicate_name_raises_integrity_error(service, repo_mock, session_mock):
    data = ProjectCreateSchema(name="dup")
    repo_mock.get_active.return_value = None

    mock_error = IntegrityError("statement", "params", "orig")
    mock_error.orig = Exception("UNIQUE constraint failed: uq_project_name")
    session_mock.commit.side_effect = mock_error

    with pytest.raises(ResourceAlreadyExistsError) as exc_info:
        service.create_project(data)

    assert exc_info.value.resource_type.value == "Project"
    assert exc_info.value.field == "name"
    assert "project with the name 'dup' already exists" in str(exc_info.value).lower()
    repo_mock.add.assert_called_once()
    session_mock.rollback.assert_called_once()
    session_mock.commit.assert_called_once()


def test_create_project_single_active_constraint_violation(service, repo_mock, session_mock):
    data = ProjectCreateSchema(name="second_active")
    existing_active = make_project(name="first_active", active=True)
    repo_mock.get_active.return_value = existing_active

    # _activate_project() should deactivate the existing one first
    result = service.create_project(data)

    assert result.active is True
    assert existing_active.active is False
    session_mock.commit.assert_called_once()


def test_get_project_success(service, repo_mock):
    pid = uuid.uuid4()
    project = make_project(project_id=pid)
    repo_mock.get_by_id.return_value = project

    result = service.get_project(pid)
    assert isinstance(result, ProjectSchema)
    assert result.id == pid
    assert result.active is False
    repo_mock.get_by_id.assert_called_once_with(pid)


def test_get_project_not_found(service, repo_mock):
    pid = uuid.uuid4()
    repo_mock.get_by_id.return_value = None
    with pytest.raises(ResourceNotFoundError):
        service.get_project(pid)


def test_list_projects(service, repo_mock):
    p1, p2 = make_project(), make_project()
    repo_mock.list_with_pagination.return_value = ([p1, p2], 2)

    result = service.list_projects(offset=0, limit=20)

    assert len(result.projects) == 2
    ids = {p.id for p in result.projects}
    assert ids == {p1.id, p2.id}
    assert result.pagination.count == 2
    assert result.pagination.total == 2
    assert result.pagination.offset == 0
    assert result.pagination.limit == 20
    repo_mock.list_with_pagination.assert_called_once_with(offset=0, limit=20)


def test_list_projects_with_pagination(service, repo_mock):
    p1 = make_project(name="project_1")
    repo_mock.list_with_pagination.return_value = ([p1], 10)

    result = service.list_projects(offset=5, limit=1)

    assert len(result.projects) == 1
    assert result.projects[0].id == p1.id
    assert result.pagination.count == 1
    assert result.pagination.total == 10
    assert result.pagination.offset == 5
    assert result.pagination.limit == 1
    repo_mock.list_with_pagination.assert_called_once_with(offset=5, limit=1)


def test_list_projects_empty(service, repo_mock):
    repo_mock.list_with_pagination.return_value = ([], 0)

    result = service.list_projects()

    assert len(result.projects) == 0
    assert result.pagination.count == 0
    assert result.pagination.total == 0
    assert result.pagination.offset == 0
    assert result.pagination.limit == 20
    repo_mock.list_with_pagination.assert_called_once_with(offset=0, limit=20)


def test_update_project_success(service, repo_mock, session_mock):
    pid = uuid.uuid4()
    existing = make_project(project_id=pid, name="old")
    repo_mock.get_by_id.return_value = existing
    repo_mock.update.return_value = existing

    data = ProjectUpdateSchema(name="new", active=existing.active)
    updated = service.update_project(pid, data)

    assert updated.name == "new"
    assert updated.active is False
    session_mock.commit.assert_called_once()
    repo_mock.update.assert_called_once()


def test_update_project_duplicate_name_raises_integrity_error(service, repo_mock, session_mock):
    pid = uuid.uuid4()
    existing = make_project(project_id=pid, name="old")
    repo_mock.get_by_id.return_value = existing

    mock_error = IntegrityError("statement", "params", "orig")
    mock_error.orig = Exception("UNIQUE constraint failed: uq_project_name")
    session_mock.commit.side_effect = mock_error

    with pytest.raises(ResourceAlreadyExistsError) as exc_info:
        service.update_project(pid, ProjectUpdateSchema(name="other", active=existing.active))

    assert exc_info.value.resource_type.value == "Project"
    assert exc_info.value.field == "name"
    session_mock.rollback.assert_called_once()


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
    assert result.active is True


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
    repo_mock.delete.assert_called_once_with(pid)
    session_mock.commit.assert_called_once()


def test_delete_project_not_found(service, repo_mock):
    repo_mock.get_by_id.return_value = None
    with pytest.raises(ResourceNotFoundError):
        service.delete_project(uuid.uuid4())


def test_get_pipeline_config_project_not_found(service, repo_mock):
    pid = uuid.uuid4()
    repo_mock.get_by_id.return_value = None
    with pytest.raises(ResourceNotFoundError):
        service.get_pipeline_config(pid)


def test_pipeline_config_with_active_source(service, repo_mock):
    pid = uuid.uuid4()
    source_active = make_active_source()
    project_active = make_project(project_id=pid, active=True, sources=[source_active])
    repo_mock.get_by_id.return_value = project_active

    cfg = service.get_pipeline_config(pid)

    assert isinstance(cfg, PipelineConfig)
    assert cfg.project_id == pid
    assert cfg.reader is not None
    assert cfg.reader.source_type == "usb_camera"
    assert cfg.processor is None
    assert cfg.writer is None


def test_pipeline_config_without_active_source(service, repo_mock):
    pid = uuid.uuid4()
    inactive_source = SimpleNamespace(
        id=uuid.uuid4(),
        active=False,
        config={"source_type": "usb_camera"},
    )
    project_active = make_project(project_id=pid, active=True, sources=[inactive_source])
    repo_mock.get_by_id.return_value = project_active

    cfg = service.get_pipeline_config(pid)

    assert isinstance(cfg, PipelineConfig)
    assert cfg.project_id == pid
    assert cfg.reader is None
    assert cfg.processor is None
    assert cfg.writer is None


def test_pipeline_config_with_source_and_sink(service, repo_mock):
    pid = uuid.uuid4()
    source_connected = make_active_source()
    active_sink = make_sink(active=True)
    project_active = make_project(
        project_id=pid,
        active=True,
        sources=[source_connected],
        sinks=[active_sink],
    )
    repo_mock.get_by_id.return_value = project_active

    cfg = service.get_pipeline_config(pid)

    assert isinstance(cfg, PipelineConfig)
    assert cfg.project_id == pid
    assert cfg.reader is not None
    assert cfg.reader.source_type == "usb_camera"
    assert cfg.processor is None
    assert cfg.writer is not None
    assert cfg.writer.sink_type == "mqtt"


def test_pipeline_config_with_source_sink_and_model(service, repo_mock):
    pid = uuid.uuid4()
    source_connected = make_active_source()
    active_sink = make_sink(active=True)
    active_model = make_model(project_id=pid, active=True)
    project_active = make_project(
        project_id=pid,
        active=True,
        sources=[source_connected],
        sinks=[active_sink],
        processors=[active_model],
    )
    repo_mock.get_by_id.return_value = project_active

    cfg = service.get_pipeline_config(pid)

    assert isinstance(cfg, PipelineConfig)
    assert cfg.project_id == pid
    assert cfg.reader is not None
    assert cfg.reader.source_type == "usb_camera"
    assert cfg.processor is not None
    assert cfg.processor.model_type == "matcher"
    assert cfg.writer is not None
    assert cfg.writer.sink_type == "mqtt"


def test_active_pipeline_config_none(service, repo_mock):
    repo_mock.get_active.return_value = None
    cfg = service.get_active_pipeline_config()
    assert cfg is None


def test_active_pipeline_config_success(service, repo_mock):
    source_active = make_active_source()
    project_active = make_project(active=True, sources=[source_active])
    repo_mock.get_active.return_value = project_active
    repo_mock.get_by_id.return_value = project_active

    cfg = service.get_active_pipeline_config()

    assert isinstance(cfg, PipelineConfig)
    assert cfg.project_id == project_active.id
    assert cfg.reader is not None
    assert cfg.reader.source_type == "usb_camera"


def test_create_emits_activation_event(service, repo_mock, dispatcher_mock):
    data = ProjectCreateSchema(name="alpha")
    repo_mock.get_active.return_value = None

    service.create_project(data)

    assert dispatcher_mock.dispatch.call_count == 1
    ev = dispatcher_mock.dispatch.call_args_list[0].args[0]
    assert isinstance(ev, ProjectActivationEvent)


def test_set_active_emits_activation_and_deactivation_events(service, repo_mock, dispatcher_mock, session_mock):
    project_previous_active = make_project(active=True)
    project_target = make_project(active=False)
    repo_mock.get_by_id.return_value = project_target
    repo_mock.get_active.return_value = project_previous_active

    service.set_active_project(project_target.id)

    assert dispatcher_mock.dispatch.call_count == 2
    event_1 = dispatcher_mock.dispatch.call_args_list[0].args[0]
    event_2 = dispatcher_mock.dispatch.call_args_list[1].args[0]
    assert isinstance(event_1, ProjectDeactivationEvent)
    assert event_1.project_id == project_previous_active.id
    assert isinstance(event_2, ProjectActivationEvent)
    assert event_2.project_id == project_target.id


def test_delete_active_emits_deactivation_event(service, repo_mock, dispatcher_mock):
    project_active = make_project(active=True)
    repo_mock.get_by_id.return_value = project_active

    service.delete_project(project_active.id)

    assert dispatcher_mock.dispatch.call_count == 1
    ev = dispatcher_mock.dispatch.call_args_list[0].args[0]
    assert isinstance(ev, ProjectDeactivationEvent)
    assert ev.project_id == project_active.id


def test_delete_inactive_emits_no_event(service, repo_mock, dispatcher_mock):
    project_inactive = make_project(active=False)
    repo_mock.get_by_id.return_value = project_inactive

    service.delete_project(project_inactive.id)

    dispatcher_mock.dispatch.assert_not_called()


def test_update_activate_emits_activation_event(service, repo_mock, dispatcher_mock):
    project_inactive = make_project(active=False)
    repo_mock.get_by_id.return_value = project_inactive
    repo_mock.get_active.return_value = None
    repo_mock.update.return_value = project_inactive

    service.update_project(project_inactive.id, ProjectUpdateSchema(active=True))

    assert dispatcher_mock.dispatch.call_count == 1
    ev = dispatcher_mock.dispatch.call_args_list[0].args[0]
    assert isinstance(ev, ProjectActivationEvent)
    assert ev.project_id == project_inactive.id


def test_update_deactivate_emits_deactivation_event(service, repo_mock, dispatcher_mock):
    project_active = make_project(active=True)
    repo_mock.get_by_id.return_value = project_active
    repo_mock.update.return_value = project_active

    service.update_project(project_active.id, ProjectUpdateSchema(active=False))

    assert dispatcher_mock.dispatch.call_count == 1
    ev = dispatcher_mock.dispatch.call_args_list[0].args[0]
    assert isinstance(ev, ProjectDeactivationEvent)
    assert ev.project_id == project_active.id


def test_update_activate_replaces_existing_active_emits_two_events(service, repo_mock, dispatcher_mock):
    project_current_active = make_project(active=True)
    project_target = make_project(active=False)
    repo_mock.get_by_id.return_value = project_target
    repo_mock.get_active.return_value = project_current_active
    repo_mock.update.return_value = project_target

    service.update_project(project_target.id, ProjectUpdateSchema(active=True))

    assert dispatcher_mock.dispatch.call_count == 2
    event_1 = dispatcher_mock.dispatch.call_args_list[0].args[0]
    event_2 = dispatcher_mock.dispatch.call_args_list[1].args[0]
    assert isinstance(event_1, ProjectDeactivationEvent)
    assert event_1.project_id == project_current_active.id
    assert isinstance(event_2, ProjectActivationEvent)
    assert event_2.project_id == project_target.id
