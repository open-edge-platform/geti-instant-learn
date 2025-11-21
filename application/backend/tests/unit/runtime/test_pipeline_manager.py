#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from queue import Queue
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from domain.dispatcher import (
    ComponentConfigChangeEvent,
    ConfigChangeDispatcher,
    ProjectActivationEvent,
    ProjectDeactivationEvent,
)
from runtime.core.components.schemas.pipeline import PipelineConfig
from runtime.errors import PipelineNotActiveError, PipelineProjectMismatchError
from runtime.pipeline_manager import PipelineManager


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


@pytest.fixture
def mock_component_factory():
    """Factory mock with pre-configured source, processor, and sink mocks."""
    mock_factory = Mock()
    mock_source = Mock()
    mock_processor = Mock()
    mock_sink = Mock()
    mock_factory.create_source.return_value = mock_source
    mock_factory.create_processor.return_value = mock_processor
    mock_factory.create_sink.return_value = mock_sink
    return mock_factory


class TestPipelineManager:
    def test_start_with_active_project_starts_pipeline_and_subscribes(
        self, dispatcher, session_factory, pipeline_cfg, mock_component_factory
    ):
        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.Pipeline") as pipeline_cls,
            patch("runtime.pipeline_manager.FrameBroadcaster"),
        ):
            svc_inst = svc_cls.return_value
            svc_inst.get_active_pipeline_config.return_value = pipeline_cfg

            # Configure the mock Pipeline to support method chaining
            pipeline_inst = pipeline_cls.return_value
            pipeline_inst.set_source.return_value = pipeline_inst
            pipeline_inst.set_processor.return_value = pipeline_inst
            pipeline_inst.set_sink.return_value = pipeline_inst

            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
            mgr.start()

            svc_inst.get_active_pipeline_config.assert_called_once()
            mock_component_factory.create_source.assert_called_once_with(pipeline_cfg.project_id)
            mock_component_factory.create_processor.assert_called_once_with(pipeline_cfg.project_id)
            mock_component_factory.create_sink.assert_called_once_with(pipeline_cfg.project_id)

            # Pipeline is called with project_id and two FrameBroadcasters
            pipeline_cls.assert_called_once()
            call_args = pipeline_cls.call_args.args
            assert call_args[0] == pipeline_cfg.project_id
            assert len(call_args) == 3  # project_id + 2 broadcasters

            # Check fluent API calls
            pipeline_inst.set_source.assert_called_once()
            pipeline_inst.set_processor.assert_called_once()
            pipeline_inst.set_sink.assert_called_once()

            pipeline_inst.start.assert_called_once()
            assert dispatcher._listeners == [mgr.on_config_change]

    def test_start_without_active_project_only_subscribes(self, dispatcher, session_factory):
        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.Pipeline") as pipeline_cls,
        ):
            svc_inst = svc_cls.return_value
            svc_inst.get_active_pipeline_config.return_value = None

            mgr = PipelineManager(dispatcher, session_factory)
            mgr.start()

            svc_inst.get_active_pipeline_config.assert_called_once()
            pipeline_cls.assert_not_called()
            assert mgr._pipeline is None
            assert dispatcher._listeners == [mgr.on_config_change]

    def test_on_activation_event_starts_new_pipeline(self, dispatcher, session_factory, mock_component_factory):
        with (
            patch("runtime.pipeline_manager.Pipeline") as pipeline_cls,
            patch("runtime.pipeline_manager.FrameBroadcaster"),
        ):
            pid = uuid4()

            # Configure the mock Pipeline to support method chaining
            pipeline_inst = pipeline_cls.return_value
            pipeline_inst.set_source.return_value = pipeline_inst
            pipeline_inst.set_processor.return_value = pipeline_inst
            pipeline_inst.set_sink.return_value = pipeline_inst

            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
            ev = ProjectActivationEvent(project_id=pid)
            mgr.on_config_change(ev)

            mock_component_factory.create_source.assert_called_once_with(pid)
            mock_component_factory.create_processor.assert_called_once_with(pid)
            mock_component_factory.create_sink.assert_called_once_with(pid)

            # Pipeline is called with project_id and two FrameBroadcasters
            pipeline_cls.assert_called_once()
            call_args = pipeline_cls.call_args.args
            assert call_args[0] == pid
            assert len(call_args) == 3  # project_id + 2 broadcasters

            # Check fluent API calls
            pipeline_inst.set_source.assert_called_once()
            pipeline_inst.set_processor.assert_called_once()
            pipeline_inst.set_sink.assert_called_once()

            pipeline_inst.start.assert_called_once()
            assert mgr._pipeline == pipeline_inst

    def test_on_activation_replaces_existing_pipeline(self, dispatcher, session_factory, mock_component_factory):
        with (
            patch("runtime.pipeline_manager.Pipeline") as pipeline_cls,
            patch("runtime.pipeline_manager.FrameBroadcaster"),
        ):
            # Existing pipeline
            old_pipeline = Mock()
            pid_new = uuid4()

            # Configure the mock Pipeline to support method chaining
            pipeline_inst = pipeline_cls.return_value
            pipeline_inst.set_source.return_value = pipeline_inst
            pipeline_inst.set_processor.return_value = pipeline_inst
            pipeline_inst.set_sink.return_value = pipeline_inst

            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
            mgr._pipeline = old_pipeline

            ev = ProjectActivationEvent(project_id=pid_new)
            mgr.on_config_change(ev)

            old_pipeline.stop.assert_called_once()
            mock_component_factory.create_source.assert_called_once_with(pid_new)
            mock_component_factory.create_processor.assert_called_once_with(pid_new)
            mock_component_factory.create_sink.assert_called_once_with(pid_new)

            # Pipeline is called with project_id and two FrameBroadcasters
            pipeline_cls.assert_called_once()
            call_args = pipeline_cls.call_args.args
            assert call_args[0] == pid_new
            assert len(call_args) == 3  # project_id + 2 broadcasters

            pipeline_inst.set_source.assert_called_once()
            pipeline_inst.set_processor.assert_called_once()
            pipeline_inst.set_sink.assert_called_once()
            pipeline_inst.start.assert_called_once()
            assert mgr._pipeline == pipeline_inst

    def test_on_deactivation_stops_matching_pipeline(self, dispatcher, session_factory):
        with patch("runtime.pipeline_manager.ProjectService"), patch("runtime.pipeline_manager.Pipeline"):
            pid = uuid4()
            running = Mock()
            running.project_id = pid

            mgr = PipelineManager(dispatcher, session_factory)
            mgr._pipeline = running

            ev = ProjectDeactivationEvent(project_id=pid)
            mgr.on_config_change(ev)

            running.stop.assert_called_once()
            assert mgr._pipeline is None

    def test_on_deactivation_ignores_non_matching_pipeline(self, dispatcher, session_factory):
        with patch("runtime.pipeline_manager.ProjectService"), patch("runtime.pipeline_manager.Pipeline"):
            running = Mock()
            running.project_id = uuid4()
            mgr = PipelineManager(dispatcher, session_factory)
            mgr._pipeline = running

            ev = ProjectDeactivationEvent(project_id=uuid4())
            mgr.on_config_change(ev)

            running.stop.assert_not_called()
            assert mgr._pipeline is running

    def test_on_component_update_applies_config_for_matching_project(
        self, dispatcher, session_factory, mock_component_factory
    ):
        with patch("runtime.pipeline_manager.Pipeline"):
            pid = uuid4()
            running = Mock()
            running.project_id = pid

            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
            mgr._pipeline = running

            ev = ComponentConfigChangeEvent(project_id=pid, component_type="source", component_id="abc")
            mgr.on_config_change(ev)

            mock_component_factory.create_source.assert_called_once_with(pid)
            running.set_source.assert_called_once()

    def test_on_component_update_ignores_mismatch(self, dispatcher, session_factory):
        with patch("runtime.pipeline_manager.Pipeline"):
            pid_running = uuid4()
            pid_event = uuid4()
            running = Mock()
            running.project_id = pid_running

            mgr = PipelineManager(dispatcher, session_factory)
            mgr._pipeline = running

            ev = ComponentConfigChangeEvent(project_id=pid_event, component_type="source", component_id="abc")
            mgr.on_config_change(ev)

            running.set_source.assert_not_called()

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

    def test_register_inbound_consumer_success(self, dispatcher, session_factory):
        with patch("runtime.pipeline_manager.ProjectService"), patch("runtime.pipeline_manager.Pipeline"):
            pid = uuid4()
            mock_pipeline = Mock()
            mock_pipeline.project_id = pid
            mock_queue = Queue()
            mock_pipeline.register_inbound_consumer.return_value = mock_queue

            mgr = PipelineManager(dispatcher, session_factory)
            mgr._pipeline = mock_pipeline

            result = mgr.register_inbound_consumer(pid)

            assert result == mock_queue
            mock_pipeline.register_inbound_consumer.assert_called_once()

    def test_register_inbound_consumer_no_pipeline(self, dispatcher, session_factory):
        mgr = PipelineManager(dispatcher, session_factory)
        mgr._pipeline = None

        with pytest.raises(PipelineNotActiveError, match="No active pipeline to register inbound consumer"):
            mgr.register_inbound_consumer(uuid4())

    def test_register_inbound_consumer_project_mismatch(self, dispatcher, session_factory):
        with patch("runtime.pipeline_manager.ProjectService"), patch("runtime.pipeline_manager.Pipeline"):
            pid_running = uuid4()
            pid_requested = uuid4()

            mock_pipeline = Mock()
            mock_pipeline.project_id = pid_running

            mgr = PipelineManager(dispatcher, session_factory)
            mgr._pipeline = mock_pipeline

            with pytest.raises(PipelineProjectMismatchError, match="does not match the active pipeline's project ID"):
                mgr.register_inbound_consumer(pid_requested)

    def test_unregister_inbound_consumer_success(self, dispatcher, session_factory):
        with patch("runtime.pipeline_manager.ProjectService"), patch("runtime.pipeline_manager.Pipeline"):
            pid = uuid4()
            mock_pipeline = Mock()
            mock_pipeline.project_id = pid
            mock_queue = Queue()

            mgr = PipelineManager(dispatcher, session_factory)
            mgr._pipeline = mock_pipeline

            mgr.unregister_inbound_consumer(pid, mock_queue)

            mock_pipeline.unregister_inbound_consumer.assert_called_once_with(mock_queue)

    def test_unregister_inbound_consumer_no_pipeline(self, dispatcher, session_factory):
        mgr = PipelineManager(dispatcher, session_factory)
        mgr._pipeline = None

        with pytest.raises(PipelineNotActiveError, match="No active pipeline to unregister inbound consumer from"):
            mgr.unregister_inbound_consumer(uuid4(), Queue())

    def test_unregister_inbound_consumer_project_mismatch(self, dispatcher, session_factory):
        with patch("runtime.pipeline_manager.ProjectService"), patch("runtime.pipeline_manager.Pipeline"):
            pid_running = uuid4()
            pid_requested = uuid4()

            mock_pipeline = Mock()
            mock_pipeline.project_id = pid_running

            mgr = PipelineManager(dispatcher, session_factory)
            mgr._pipeline = mock_pipeline

            with pytest.raises(PipelineProjectMismatchError, match="does not match the active pipeline's project ID"):
                mgr.unregister_inbound_consumer(pid_requested, Queue())
