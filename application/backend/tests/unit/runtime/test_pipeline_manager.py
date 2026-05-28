#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from domain.dispatcher import (
    ComponentConfigChangeEvent,
    ComponentType,
    ConfigChangeDispatcher,
    ProjectActivationEvent,
    ProjectDeactivationEvent,
)
from domain.services.schemas.model_status import ModelStatus, ModelStatusErrorType
from domain.services.schemas.pipeline import PipelineConfig
from runtime.errors import PipelineNotActiveError, PipelineProjectMismatchError, PipelineReloadInProgressError
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
            patch("runtime.pipeline_manager.FrameRepository") as repo_cls,
            patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls,
            patch.object(PipelineManager, "_refresh_visualization_info", return_value=None),
        ):
            svc_inst = svc_cls.return_value
            svc_inst.get_active_pipeline_config.return_value = pipeline_cfg
            svc_inst.get_pipeline_config.return_value = pipeline_cfg
            repo_inst = repo_cls.return_value
            batch_svc_cls.return_value.build.return_value = None

            # Configure the mock Pipeline to support method chaining
            pipeline_inst = pipeline_cls.return_value
            pipeline_inst.set_source.return_value = pipeline_inst
            pipeline_inst.set_processor.return_value = pipeline_inst
            pipeline_inst.set_sink.return_value = pipeline_inst

            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
            mgr.start()

            svc_inst.get_active_pipeline_config.assert_called_once()
            mock_component_factory.create_source.assert_called_once_with(pipeline_cfg.reader)
            mock_component_factory.create_processor.assert_called_once_with(pipeline_cfg, None)
            mock_component_factory.create_sink.assert_called_once_with(pipeline_cfg.writer)

            # Pipeline is called with project_id and two FrameBroadcasters
            pipeline_cls.assert_called_once()
            call_args = pipeline_cls.call_args.args
            assert call_args[0] == pipeline_cfg.project_id
            assert call_args[1] == repo_inst
            assert len(call_args) == 4  # project_id + repo + 2 broadcasters

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
            patch("runtime.pipeline_manager.FrameRepository"),
            patch("runtime.pipeline_manager.ReferenceBatchService"),
        ):
            svc_inst = svc_cls.return_value
            svc_inst.get_active_pipeline_config.return_value = None

            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            mgr.start()

            svc_inst.get_active_pipeline_config.assert_called_once()
            pipeline_cls.assert_not_called()
            assert mgr._pipeline is None
            assert dispatcher._listeners == [mgr.on_config_change]

    def test_start_with_active_project_stores_error_and_keeps_running(
        self, dispatcher, session_factory, pipeline_cfg, mock_component_factory
    ):
        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls,
        ):
            svc_inst = svc_cls.return_value
            svc_inst.get_active_pipeline_config.return_value = pipeline_cfg
            svc_inst.get_pipeline_config.return_value = pipeline_cfg
            batch_svc_cls.return_value.build.return_value = None
            mock_component_factory.create_processor.side_effect = RuntimeError("boom")

            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)

            mgr.start()

            status = mgr.get_model_status()
            assert mgr._pipeline is None
            assert status.status == ModelStatus.ERROR
            assert status.error_type == ModelStatusErrorType.LOAD_FAILED
            assert status.error_message is not None
            assert dispatcher._listeners == [mgr.on_config_change]

    def test_on_activation_event_starts_new_pipeline(self, dispatcher, session_factory, mock_component_factory):
        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.Pipeline") as pipeline_cls,
            patch("runtime.pipeline_manager.FrameBroadcaster"),
            patch("runtime.pipeline_manager.FrameRepository") as repo_cls,
            patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls,
            patch.object(PipelineManager, "_refresh_visualization_info", return_value=None),
        ):
            pid = uuid4()
            cfg = PipelineConfig(project_id=pid)
            svc_inst = svc_cls.return_value
            svc_inst.get_pipeline_config.return_value = cfg
            repo_inst = repo_cls.return_value
            batch_svc_cls.return_value.build.return_value = None

            # Configure the mock Pipeline to support method chaining
            pipeline_inst = pipeline_cls.return_value
            pipeline_inst.set_source.return_value = pipeline_inst
            pipeline_inst.set_processor.return_value = pipeline_inst
            pipeline_inst.set_sink.return_value = pipeline_inst

            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
            ev = ProjectActivationEvent(project_id=pid)
            mgr.on_config_change(ev)

            mock_component_factory.create_source.assert_called_once_with(cfg.reader)
            mock_component_factory.create_processor.assert_called_once_with(cfg, None)
            mock_component_factory.create_sink.assert_called_once_with(cfg.writer)

            # Pipeline is called with project_id and two FrameBroadcasters
            pipeline_cls.assert_called_once()
            call_args = pipeline_cls.call_args.args
            assert call_args[0] == pid
            assert call_args[1] == repo_inst
            assert len(call_args) == 4  # project_id + repo + 2 broadcasters

            # Check fluent API calls
            pipeline_inst.set_source.assert_called_once()
            pipeline_inst.set_processor.assert_called_once()
            pipeline_inst.set_sink.assert_called_once()

            pipeline_inst.start.assert_called_once()
            assert mgr._pipeline == pipeline_inst

    def test_on_activation_replaces_existing_pipeline(self, dispatcher, session_factory, mock_component_factory):
        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.Pipeline") as pipeline_cls,
            patch("runtime.pipeline_manager.FrameBroadcaster"),
            patch("runtime.pipeline_manager.FrameRepository"),
            patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls,
            patch.object(PipelineManager, "_refresh_visualization_info", return_value=None),
        ):
            # Existing pipeline
            old_pipeline = Mock()
            pid_new = uuid4()
            cfg = PipelineConfig(project_id=pid_new)
            svc_inst = svc_cls.return_value
            svc_inst.get_pipeline_config.return_value = cfg
            batch_svc_cls.return_value.build.return_value = None

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
            mock_component_factory.create_source.assert_called_once_with(cfg.reader)
            mock_component_factory.create_processor.assert_called_once_with(cfg, None)
            mock_component_factory.create_sink.assert_called_once_with(cfg.writer)

            # Pipeline is called with project_id and two FrameBroadcasters
            pipeline_cls.assert_called_once()
            call_args = pipeline_cls.call_args.args
            assert call_args[0] == pid_new
            assert len(call_args) == 4  # project_id + repo + 2 broadcasters

            pipeline_inst.set_source.assert_called_once()
            pipeline_inst.set_processor.assert_called_once()
            pipeline_inst.set_sink.assert_called_once()
            pipeline_inst.start.assert_called_once()
            assert mgr._pipeline == pipeline_inst

    def test_get_visualization_info_raises_when_pipeline_inactive(self, dispatcher, session_factory):
        with patch("runtime.pipeline_manager.FrameRepository"), patch("runtime.pipeline_manager.ReferenceBatchService"):
            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            with pytest.raises(PipelineNotActiveError):
                mgr.get_visualization_info(uuid4())

    def test_get_visualization_info_raises_when_project_mismatched(self, dispatcher, session_factory):
        with patch("runtime.pipeline_manager.FrameRepository"), patch("runtime.pipeline_manager.ReferenceBatchService"):
            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            running = Mock()
            running.project_id = uuid4()
            mgr._pipeline = running

            with pytest.raises(PipelineProjectMismatchError):
                mgr.get_visualization_info(uuid4())

    def test_get_visualization_info_returns_cached_value(self, dispatcher, session_factory):
        with patch("runtime.pipeline_manager.FrameRepository"), patch("runtime.pipeline_manager.ReferenceBatchService"):
            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            pid = uuid4()
            running = Mock()
            running.project_id = pid
            mgr._pipeline = running

            cached = Mock()
            mgr._visualization_info = cached

            assert mgr.get_visualization_info(pid) is cached

    def test_on_deactivation_stops_matching_pipeline(self, dispatcher, session_factory):
        with (
            patch("runtime.pipeline_manager.ProjectService"),
            patch("runtime.pipeline_manager.Pipeline"),
            patch("runtime.pipeline_manager.FrameRepository"),
            patch("runtime.pipeline_manager.ReferenceBatchService"),
        ):
            pid = uuid4()
            running = Mock()
            running.project_id = pid

            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            mgr._pipeline = running

            ev = ProjectDeactivationEvent(project_id=pid)
            mgr.on_config_change(ev)

            running.stop.assert_called_once()

    def test_on_deactivation_ignores_non_matching_pipeline(self, dispatcher, session_factory):
        with (
            patch("runtime.pipeline_manager.ProjectService"),
            patch("runtime.pipeline_manager.Pipeline"),
            patch("runtime.pipeline_manager.FrameRepository"),
            patch("runtime.pipeline_manager.ReferenceBatchService"),
        ):
            running = Mock()
            running.project_id = uuid4()
            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            mgr._pipeline = running

            ev = ProjectDeactivationEvent(project_id=uuid4())
            mgr.on_config_change(ev)

            running.stop.assert_not_called()
            assert mgr._pipeline is running

    def test_on_component_update_applies_config_for_matching_project(
        self, dispatcher, session_factory, mock_component_factory
    ):
        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.Pipeline"),
            patch("runtime.pipeline_manager.FrameRepository"),
            patch("runtime.pipeline_manager.ReferenceBatchService"),
        ):
            pid = uuid4()
            component_id = uuid4()
            cfg = PipelineConfig(project_id=pid)
            svc_inst = svc_cls.return_value
            svc_inst.get_pipeline_config.return_value = cfg

            running = Mock()
            running.project_id = pid

            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
            mgr._pipeline = running

            ev = ComponentConfigChangeEvent(
                project_id=pid, component_type=ComponentType.SOURCE, component_id=component_id
            )
            mgr.on_config_change(ev)

            mock_component_factory.create_source.assert_called_once_with(cfg.reader)
            running.set_source.assert_called_once()

    def test_on_component_update_ignores_mismatch(self, dispatcher, session_factory):
        with (
            patch("runtime.pipeline_manager.Pipeline"),
            patch("runtime.pipeline_manager.FrameRepository"),
            patch("runtime.pipeline_manager.ReferenceBatchService"),
        ):
            pid_running = uuid4()
            pid_event = uuid4()
            component_id = uuid4()
            running = Mock()
            running.project_id = pid_running

            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            mgr._pipeline = running

            ev = ComponentConfigChangeEvent(
                project_id=pid_event, component_type=ComponentType.SOURCE, component_id=component_id
            )
            mgr.on_config_change(ev)

            running.set_source.assert_not_called()

    def test_stop_stops_pipeline_if_present(self, dispatcher, session_factory):
        with patch("runtime.pipeline_manager.FrameRepository"), patch("runtime.pipeline_manager.ReferenceBatchService"):
            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            running = Mock()
            mgr._pipeline = running

            mgr.stop()

            running.stop.assert_called_once()
            assert mgr._pipeline is None

    def test_stop_no_pipeline_noop(self, dispatcher, session_factory):
        with patch("runtime.pipeline_manager.FrameRepository"), patch("runtime.pipeline_manager.ReferenceBatchService"):
            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            mgr._pipeline = None
            mgr.stop()
            assert mgr._pipeline is None


class TestPipelineManagerModelLoadingFlag:
    """Tests for the processor load status tracked around processor (re)builds."""

    def test_status_defaults_to_ready(self, dispatcher, session_factory):
        mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
        assert mgr.is_model_loading() is False
        status = mgr.get_model_status()
        assert status.status == ModelStatus.READY
        assert status.error_type is None
        assert status.error_message is None

    def test_status_set_during_processor_rebuild(self, dispatcher, session_factory, mock_component_factory):
        """While create_processor runs, the status must report loading and then return to ready."""
        with patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls:
            batch_svc_cls.return_value.build.return_value = None
            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
        # A running pipeline is required for _update_pipeline_components to do anything.
        mgr._pipeline = Mock()
        mgr._pipeline.project_id = uuid4()

        observed: list[ModelStatus] = []

        def fake_create_processor(*args, **kwargs):
            observed.append(mgr.get_model_status().status)
            return Mock()

        mock_component_factory.create_processor.side_effect = fake_create_processor

        with patch("runtime.pipeline_manager.ProjectService") as svc_cls:
            svc_cls.return_value.get_pipeline_config.return_value = PipelineConfig(project_id=mgr._pipeline.project_id)
            mgr._update_pipeline_components(mgr._pipeline.project_id, ComponentType.PROCESSOR)

        assert observed == [ModelStatus.LOADING]
        assert mgr.is_model_loading() is False
        status = mgr.get_model_status()
        assert status.status == ModelStatus.READY
        assert status.error_type is None
        assert status.error_message is None

    def test_status_set_to_generic_error_when_processor_rebuild_fails(
        self, dispatcher, session_factory, mock_component_factory
    ):
        with patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls:
            batch_svc_cls.return_value.build.return_value = None
            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
        mgr._pipeline = Mock()
        mgr._pipeline.project_id = uuid4()

        mock_component_factory.create_processor.side_effect = RuntimeError("boom")

        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            pytest.raises(RuntimeError),
        ):
            svc_cls.return_value.get_pipeline_config.return_value = PipelineConfig(project_id=mgr._pipeline.project_id)
            mgr._update_pipeline_components(mgr._pipeline.project_id, ComponentType.PROCESSOR)

        assert mgr.is_model_loading() is False
        status = mgr.get_model_status()
        assert status.status == ModelStatus.ERROR
        assert status.error_type == ModelStatusErrorType.LOAD_FAILED
        assert status.error_message is not None
        assert "Check the backend logs" in status.error_message

    def test_status_set_to_auth_error_when_processor_rebuild_hits_gated_repo(
        self, dispatcher, session_factory, mock_component_factory
    ):
        with patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls:
            batch_svc_cls.return_value.build.return_value = None
            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
        mgr._pipeline = Mock()
        mgr._pipeline.project_id = uuid4()

        auth_exc = OSError(
            "You are trying to access a gated repo. Make sure to have access to it at "
            "https://huggingface.co/facebook/sam3.1. Please log in."
        )
        mock_component_factory.create_processor.side_effect = auth_exc

        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            pytest.raises(OSError),
        ):
            svc_cls.return_value.get_pipeline_config.return_value = PipelineConfig(project_id=mgr._pipeline.project_id)
            mgr._update_pipeline_components(mgr._pipeline.project_id, ComponentType.PROCESSOR)

        status = mgr.get_model_status()
        assert status.status == ModelStatus.ERROR
        assert status.error_type == ModelStatusErrorType.AUTH_REQUIRED
        assert status.error_message is not None
        assert "hf auth login" in status.error_message

    def test_successful_rebuild_clears_previous_error(self, dispatcher, session_factory, mock_component_factory):
        with patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls:
            batch_svc_cls.return_value.build.return_value = None
            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
        mgr._pipeline = Mock()
        mgr._pipeline.project_id = uuid4()
        mgr._set_model_status(
            ModelStatus.ERROR,
            error_type=ModelStatusErrorType.LOAD_FAILED,
            error_message="old error",
        )

        with patch("runtime.pipeline_manager.ProjectService") as svc_cls:
            svc_cls.return_value.get_pipeline_config.return_value = PipelineConfig(project_id=mgr._pipeline.project_id)
            mgr._update_pipeline_components(mgr._pipeline.project_id, ComponentType.PROCESSOR)

        status = mgr.get_model_status()
        assert status.status == ModelStatus.READY
        assert status.error_type is None
        assert status.error_message is None

    def test_reload_pipeline_restarts_full_pipeline(
        self, dispatcher, session_factory, pipeline_cfg, mock_component_factory
    ):
        old_pipeline = Mock()
        old_pipeline.project_id = pipeline_cfg.project_id

        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.Pipeline") as pipeline_cls,
            patch("runtime.pipeline_manager.FrameBroadcaster"),
            patch("runtime.pipeline_manager.FrameRepository") as repo_cls,
            patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls,
            patch.object(PipelineManager, "_refresh_visualization_info", return_value=None),
        ):
            svc_cls.return_value.get_pipeline_config.return_value = pipeline_cfg
            batch_svc_cls.return_value.build.return_value = None
            repo_inst = repo_cls.return_value
            pipeline_inst = pipeline_cls.return_value
            pipeline_inst.set_source.return_value = pipeline_inst
            pipeline_inst.set_processor.return_value = pipeline_inst
            pipeline_inst.set_sink.return_value = pipeline_inst

            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
            mgr._pipeline = old_pipeline

            mgr.reload_pipeline(pipeline_cfg.project_id)

            old_pipeline.stop.assert_called_once()
            pipeline_cls.assert_called_once()
            call_args = pipeline_cls.call_args.args
            assert call_args[0] == pipeline_cfg.project_id
            assert call_args[1] == repo_inst
            pipeline_inst.start.assert_called_once()
            assert mgr._pipeline == pipeline_inst
            assert mgr.get_model_status().status == ModelStatus.READY

    def test_reload_pipeline_raises_conflict_when_loading_is_already_in_progress(
        self, dispatcher, session_factory, mock_component_factory
    ):
        mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
        mgr._set_model_status(ModelStatus.LOADING)

        with pytest.raises(PipelineReloadInProgressError):
            mgr.reload_pipeline(uuid4())
