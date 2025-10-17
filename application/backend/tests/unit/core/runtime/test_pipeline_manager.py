#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from core.runtime.dispatcher import (
    ComponentConfigChangeEvent,
    ConfigChangeDispatcher,
    ProjectActivationEvent,
    ProjectDeactivationEvent,
)
from core.runtime.pipeline_manager import PipelineManager
from core.runtime.schemas.pipeline import PipelineConfig


class FakeSessionCtx:
    """Minimal session factory context manager returning a mock session."""

    def __init__(self):
        self.session = Mock()

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeSessionFactory:
    """Callable returning a context manager compatible with 'with session_factory() as s:'."""

    def __call__(self):
        return FakeSessionCtx()


@pytest.fixture
def dispatcher():
    return ConfigChangeDispatcher()


@pytest.fixture
def session_factory():
    return FakeSessionFactory()


@pytest.fixture
def pipeline_cfg():
    return PipelineConfig(
        project_id=uuid4(),
        reader=None,
        processor=None,
        writer=None,
    )


class TestPipelineManager:
    def test_start_with_active_project_starts_pipeline_and_subscribes(self, dispatcher, session_factory, pipeline_cfg):
        active_uuid = uuid4()

        with (
            patch("core.runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("core.runtime.pipeline_manager.Pipeline") as pipeline_cls,
        ):
            svc_inst = svc_cls.return_value
            # Active project returns config
            cfg = PipelineConfig(project_id=active_uuid, reader=None, processor=None, writer=None)
            svc_inst.get_active_pipeline_config.return_value = cfg

            mgr = PipelineManager(dispatcher, session_factory)
            mgr.start()

            svc_cls.assert_called()  # service constructed
            svc_inst.get_active_pipeline_config.assert_called_once()
            pipeline_cls.assert_called_once_with(pipeline_conf=cfg)
            pipeline_cls.return_value.start.assert_called_once()
            assert dispatcher._listeners == [mgr.on_config_change]

    def test_start_without_active_project_only_subscribes(self, dispatcher, session_factory):
        with (
            patch("core.runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("core.runtime.pipeline_manager.Pipeline") as pipeline_cls,
        ):
            svc_inst = svc_cls.return_value
            svc_inst.get_active_pipeline_config.return_value = None

            mgr = PipelineManager(dispatcher, session_factory)
            mgr.start()

            svc_inst.get_active_pipeline_config.assert_called_once()
            pipeline_cls.assert_not_called()
            assert mgr._pipeline is None
            assert dispatcher._listeners == [mgr.on_config_change]

    def test_on_activation_event_starts_new_pipeline(self, dispatcher, session_factory):
        with (
            patch("core.runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("core.runtime.pipeline_manager.Pipeline") as pipeline_cls,
        ):
            svc_inst = svc_cls.return_value
            pid = uuid4()
            cfg = PipelineConfig(project_id=pid, reader=None, processor=None, writer=None)
            svc_inst.get_pipeline_config.return_value = cfg

            mgr = PipelineManager(dispatcher, session_factory)
            ev = ProjectActivationEvent(project_id=pid)  # removed str()
            mgr.on_config_change(ev)

            svc_inst.get_pipeline_config.assert_called_once_with(pid)
            pipeline_cls.assert_called_once_with(pipeline_conf=cfg)
            pipeline_cls.return_value.start.assert_called_once()
            assert mgr._pipeline == pipeline_cls.return_value

    def test_on_activation_replaces_existing_pipeline(self, dispatcher, session_factory):
        with (
            patch("core.runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("core.runtime.pipeline_manager.Pipeline") as pipeline_cls,
        ):
            # Existing pipeline
            old_pipeline = Mock()
            pid_new = uuid4()
            cfg_new = PipelineConfig(project_id=pid_new, reader=None, processor=None, writer=None)

            svc_inst = svc_cls.return_value
            svc_inst.get_pipeline_config.return_value = cfg_new

            mgr = PipelineManager(dispatcher, session_factory)
            mgr._pipeline = old_pipeline

            ev = ProjectActivationEvent(project_id=pid_new)
            mgr.on_config_change(ev)

            old_pipeline.stop.assert_called_once()
            pipeline_cls.assert_called_once_with(pipeline_conf=cfg_new)
            pipeline_cls.return_value.start.assert_called_once()
            assert mgr._pipeline == pipeline_cls.return_value

    def test_on_deactivation_stops_matching_pipeline(self, dispatcher, session_factory):
        with patch("core.runtime.pipeline_manager.ProjectService"), patch("core.runtime.pipeline_manager.Pipeline"):
            pid = uuid4()
            running = Mock()
            running.config.project_id = pid

            mgr = PipelineManager(dispatcher, session_factory)
            mgr._pipeline = running

            ev = ProjectDeactivationEvent(project_id=pid)
            mgr.on_config_change(ev)

            running.stop.assert_called_once()
            assert mgr._pipeline is None

    def test_on_deactivation_ignores_non_matching_pipeline(self, dispatcher, session_factory):
        with patch("core.runtime.pipeline_manager.ProjectService"), patch("core.runtime.pipeline_manager.Pipeline"):
            running = Mock()
            running.config.project_id = uuid4()
            mgr = PipelineManager(dispatcher, session_factory)
            mgr._pipeline = running

            ev = ProjectDeactivationEvent(project_id=uuid4())
            mgr.on_config_change(ev)

            running.stop.assert_not_called()
            assert mgr._pipeline is running

    def test_on_component_update_applies_config_for_matching_project(self, dispatcher, session_factory):
        with patch("core.runtime.pipeline_manager.ProjectService") as svc_cls:
            pid = uuid4()
            running = Mock()
            running.config.project_id = pid
            new_cfg = PipelineConfig(project_id=pid, reader=None, processor=None, writer=None)

            svc_cls.return_value.get_pipeline_config.return_value = new_cfg

            mgr = PipelineManager(dispatcher, session_factory)
            mgr._pipeline = running

            ev = ComponentConfigChangeEvent(project_id=pid, component_type="source", component_id="abc")
            mgr.on_config_change(ev)

            svc_cls.return_value.get_pipeline_config.assert_called_once_with(pid)
            running.update_config.assert_called_once_with(new_cfg)

    def test_on_component_update_ignores_mismatch(self, dispatcher, session_factory):
        with patch("core.runtime.pipeline_manager.ProjectService") as svc_cls:
            pid_running = uuid4()
            pid_event = uuid4()
            running = Mock()
            running.config.project_id = pid_running

            mgr = PipelineManager(dispatcher, session_factory)
            mgr._pipeline = running

            ev = ComponentConfigChangeEvent(project_id=pid_event, component_type="source", component_id="abc")
            mgr.on_config_change(ev)

            svc_cls.return_value.get_pipeline_config.assert_not_called()
            running.update_config.assert_not_called()

    def test_stop_stops_pipeline_if_present(self, dispatcher, session_factory):
        mgr = PipelineManager(dispatcher, session_factory)
        running = Mock()
        mgr._pipeline = running

        mgr.stop()

        running.stop.assert_called_once()
        assert mgr._pipeline is None

    def test_stop_no_pipeline_noop(self, dispatcher, session_factory):
        mgr = PipelineManager(dispatcher, session_factory)
        mgr._pipeline = None
        mgr.stop()
        assert mgr._pipeline is None
