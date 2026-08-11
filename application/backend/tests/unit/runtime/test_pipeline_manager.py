#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import threading
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

import pytest

from domain.dispatcher import (
    ComponentConfigChangeEvent,
    ComponentType,
    ConfigChangeDispatcher,
    ProjectActivationEvent,
    ProjectDeactivationEvent,
)
from domain.services.schemas.model_status import ModelStatus, ModelStatusErrorType, ModelStatusSchema
from domain.services.schemas.pipeline import PipelineConfig
from runtime.errors import PipelineNotActiveError, PipelineProjectMismatchError, PipelineReloadInProgressError
from runtime.pipeline_manager import PipelineManager, _ActiveState
from runtime.services.model_load_error import _ACCESS_REQUIRED_MESSAGE, _AUTH_REQUIRED_MESSAGE


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


def activate(mgr: PipelineManager, pipeline: Mock, project_id: UUID | None = None) -> Mock:
    """Install a fake running pipeline as the manager's active state."""
    if project_id is not None:
        pipeline.project_id = project_id
    mgr._state = _ActiveState(pipeline=pipeline, config=PipelineConfig(project_id=pipeline.project_id))
    return pipeline


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
            patch.object(PipelineManager, "_load_visualization_info", return_value=None),
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
            assert mgr._state is not None
            assert mgr._state.pipeline is pipeline_inst
            assert mgr._state.config is pipeline_cfg
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
            assert mgr._state is None
            assert dispatcher._listeners == [mgr.on_config_change]
            assert mgr.get_model_status().status is None

    def test_start_with_active_project_stores_error_and_keeps_running(
        self, dispatcher, session_factory, pipeline_cfg, mock_component_factory
    ):
        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.Pipeline") as pipeline_cls,
            patch("runtime.pipeline_manager.FrameBroadcaster"),
            patch("runtime.pipeline_manager.FrameRepository"),
            patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls,
            patch.object(PipelineManager, "_load_visualization_info", return_value=None),
        ):
            svc_inst = svc_cls.return_value
            svc_inst.get_active_pipeline_config.return_value = pipeline_cfg
            svc_inst.get_pipeline_config.return_value = pipeline_cfg
            batch_svc_cls.return_value.build.return_value = None
            # First call raises; second call is the passthrough fallback with reference_batch=None
            mock_component_factory.create_processor.side_effect = [RuntimeError("boom"), Mock()]

            pipeline_inst = pipeline_cls.return_value
            pipeline_inst.set_source.return_value = pipeline_inst
            pipeline_inst.set_processor.return_value = pipeline_inst
            pipeline_inst.set_sink.return_value = pipeline_inst

            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
            mgr.start()

            status = mgr.get_model_status()
            assert mgr._state is not None  # pipeline started with passthrough processor
            assert status.status == ModelStatus.ERROR
            assert status.error_type == ModelStatusErrorType.LOAD_FAILED
            assert status.error_message == "boom"
            assert dispatcher._listeners == [mgr.on_config_change]
            # Passthrough called with reference_batch=None
            mock_component_factory.create_processor.assert_called_with(pipeline_cfg, None)

    def test_start_reraises_when_even_passthrough_fails(
        self, dispatcher, session_factory, pipeline_cfg, mock_component_factory
    ):
        """A failing passthrough fallback must keep the original error details."""
        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.Pipeline"),
            patch("runtime.pipeline_manager.FrameBroadcaster"),
            patch("runtime.pipeline_manager.FrameRepository"),
            patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls,
        ):
            svc_cls.return_value.get_pipeline_config.return_value = pipeline_cfg
            batch_svc_cls.return_value.build.return_value = None
            mock_component_factory.create_processor.side_effect = [RuntimeError("boom"), RuntimeError("no fallback")]

            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
            with pytest.raises(RuntimeError):
                mgr._build_and_start_pipeline(pipeline_cfg.project_id)

            assert mgr._state is None
            status = mgr.get_model_status()
            assert status.status == ModelStatus.ERROR
            assert status.error_message == "boom"

    def test_on_activation_event_starts_new_pipeline(self, dispatcher, session_factory, mock_component_factory):
        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.Pipeline") as pipeline_cls,
            patch("runtime.pipeline_manager.FrameBroadcaster"),
            patch("runtime.pipeline_manager.FrameRepository") as repo_cls,
            patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls,
            patch.object(PipelineManager, "_load_visualization_info", return_value=None),
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
            assert mgr._state.pipeline == pipeline_inst

    def test_on_activation_replaces_existing_pipeline(self, dispatcher, session_factory, mock_component_factory):
        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.Pipeline") as pipeline_cls,
            patch("runtime.pipeline_manager.FrameBroadcaster"),
            patch("runtime.pipeline_manager.FrameRepository"),
            patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls,
            patch.object(PipelineManager, "_load_visualization_info", return_value=None),
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
            activate(mgr, old_pipeline, project_id=uuid4())

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
            assert mgr._state.pipeline == pipeline_inst

    def test_get_visualization_info_raises_when_pipeline_inactive(self, dispatcher, session_factory):
        with patch("runtime.pipeline_manager.FrameRepository"), patch("runtime.pipeline_manager.ReferenceBatchService"):
            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            with pytest.raises(PipelineNotActiveError):
                mgr.get_visualization_info(uuid4())

    def test_get_visualization_info_raises_when_project_mismatched(self, dispatcher, session_factory):
        with patch("runtime.pipeline_manager.FrameRepository"), patch("runtime.pipeline_manager.ReferenceBatchService"):
            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            activate(mgr, Mock(), project_id=uuid4())

            with pytest.raises(PipelineProjectMismatchError):
                mgr.get_visualization_info(uuid4())

    def test_get_visualization_info_returns_cached_value(self, dispatcher, session_factory):
        with patch("runtime.pipeline_manager.FrameRepository"), patch("runtime.pipeline_manager.ReferenceBatchService"):
            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            pid = uuid4()
            activate(mgr, Mock(), project_id=pid)

            cached = Mock()
            mgr._state = _ActiveState(pipeline=mgr._state.pipeline, config=mgr._state.config, visualization_info=cached)

            assert mgr.get_visualization_info(pid) is cached

    @pytest.mark.parametrize("accessor", ["seek", "get_frame_index", "list_frames", "capture_frame"])
    def test_accessors_raise_when_pipeline_inactive(self, dispatcher, session_factory, accessor):
        """The navigation accessors must not observe a torn-down pipeline."""
        with patch("runtime.pipeline_manager.FrameRepository"), patch("runtime.pipeline_manager.ReferenceBatchService"):
            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            args = (0,) if accessor == "seek" else ()
            with pytest.raises(PipelineNotActiveError):
                getattr(mgr, accessor)(uuid4(), *args)

    @pytest.mark.parametrize("accessor", ["seek", "get_frame_index", "list_frames", "capture_frame"])
    def test_accessors_raise_on_project_mismatch(self, dispatcher, session_factory, accessor):
        with patch("runtime.pipeline_manager.FrameRepository"), patch("runtime.pipeline_manager.ReferenceBatchService"):
            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            activate(mgr, Mock(), project_id=uuid4())
            args = (0,) if accessor == "seek" else ()
            with pytest.raises(PipelineProjectMismatchError):
                getattr(mgr, accessor)(uuid4(), *args)

    def test_on_deactivation_stops_matching_pipeline(self, dispatcher, session_factory):
        with (
            patch("runtime.pipeline_manager.ProjectService"),
            patch("runtime.pipeline_manager.Pipeline"),
            patch("runtime.pipeline_manager.FrameRepository"),
            patch("runtime.pipeline_manager.ReferenceBatchService"),
        ):
            pid = uuid4()
            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            running = activate(mgr, Mock(), project_id=pid)

            ev = ProjectDeactivationEvent(project_id=pid)
            mgr.on_config_change(ev)

            running.stop.assert_called_once()
            assert mgr._state is None

    def test_on_deactivation_ignores_non_matching_pipeline(self, dispatcher, session_factory):
        with (
            patch("runtime.pipeline_manager.ProjectService"),
            patch("runtime.pipeline_manager.Pipeline"),
            patch("runtime.pipeline_manager.FrameRepository"),
            patch("runtime.pipeline_manager.ReferenceBatchService"),
        ):
            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            running = activate(mgr, Mock(), project_id=uuid4())

            ev = ProjectDeactivationEvent(project_id=uuid4())
            mgr.on_config_change(ev)

            running.stop.assert_not_called()
            assert mgr._state.pipeline is running

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

            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
            running = activate(mgr, Mock(), project_id=pid)

            ev = ComponentConfigChangeEvent(
                project_id=pid, component_type=ComponentType.SOURCE, component_id=component_id
            )
            mgr.on_config_change(ev)

            mock_component_factory.create_source.assert_called_once_with(cfg.reader)
            running.set_source.assert_called_once()
            assert mgr._state.config is cfg

    def test_on_component_update_ignores_mismatch(self, dispatcher, session_factory):
        with (
            patch("runtime.pipeline_manager.Pipeline"),
            patch("runtime.pipeline_manager.FrameRepository"),
            patch("runtime.pipeline_manager.ReferenceBatchService"),
        ):
            component_id = uuid4()
            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            running = activate(mgr, Mock(), project_id=uuid4())

            ev = ComponentConfigChangeEvent(
                project_id=uuid4(), component_type=ComponentType.SOURCE, component_id=component_id
            )
            mgr.on_config_change(ev)

            running.set_source.assert_not_called()

    def test_stop_stops_pipeline_if_present(self, dispatcher, session_factory):
        with patch("runtime.pipeline_manager.FrameRepository"), patch("runtime.pipeline_manager.ReferenceBatchService"):
            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            running = activate(mgr, Mock(), project_id=uuid4())

            mgr.stop()

            running.stop.assert_called_once()
            assert mgr._state is None

    def test_stop_no_pipeline_noop(self, dispatcher, session_factory):
        with patch("runtime.pipeline_manager.FrameRepository"), patch("runtime.pipeline_manager.ReferenceBatchService"):
            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            mgr.stop()
            assert mgr._state is None


class TestPipelineManagerModelLoadingFlag:
    """Tests for the processor load status tracked around processor (re)builds."""

    def test_status_defaults_to_initializing(self, dispatcher, session_factory):
        mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
        assert mgr.is_model_loading() is False
        status = mgr.get_model_status()
        assert status.status is None
        assert status.error_type is None
        assert status.error_message is None

    def test_status_set_during_processor_rebuild(self, dispatcher, session_factory, mock_component_factory):
        """While create_processor runs, the status must report loading and then return to ready."""
        with patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls:
            batch_svc_cls.return_value.build.return_value = None
            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
        # A running pipeline is required for _update_pipeline_components to do anything.
        pid = uuid4()
        activate(mgr, Mock(), project_id=pid)

        observed: list[ModelStatus] = []

        def fake_create_processor(*args, **kwargs):
            observed.append(mgr.get_model_status().status)
            return Mock()

        mock_component_factory.create_processor.side_effect = fake_create_processor

        with patch("runtime.pipeline_manager.ProjectService") as svc_cls:
            svc_cls.return_value.get_pipeline_config.return_value = PipelineConfig(project_id=pid)
            mgr._update_pipeline_components(pid, ComponentType.PROCESSOR)

        assert observed == [ModelStatus.LOADING]
        assert mgr.is_model_loading() is False
        status = mgr.get_model_status()
        assert status.status == ModelStatus.READY
        assert status.error_type is None
        assert status.error_message is None

    def test_ready_is_published_only_after_the_processor_is_installed(
        self, dispatcher, session_factory, mock_component_factory
    ):
        """READY must not be visible while the built processor is not yet swapped in."""
        with patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls:
            batch_svc_cls.return_value.build.return_value = None
            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
        pid = uuid4()
        running = activate(mgr, Mock(), project_id=pid)

        observed: list[ModelStatus] = []
        running.set_processor.side_effect = lambda *a, **kw: observed.append(mgr.get_model_status().status)

        with patch("runtime.pipeline_manager.ProjectService") as svc_cls:
            svc_cls.return_value.get_pipeline_config.return_value = PipelineConfig(project_id=pid)
            mgr._update_pipeline_components(pid, ComponentType.PROCESSOR)

        assert observed == [ModelStatus.LOADING]
        assert mgr.get_model_status().status == ModelStatus.READY

    def test_source_update_leaves_model_status_untouched(self, dispatcher, session_factory, mock_component_factory):
        """The model status describes the processor only."""
        with patch("runtime.pipeline_manager.ReferenceBatchService"):
            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
        pid = uuid4()
        activate(mgr, Mock(), project_id=pid)
        mgr._store_model_status(ModelStatusSchema(status=ModelStatus.READY))

        with patch("runtime.pipeline_manager.ProjectService") as svc_cls:
            svc_cls.return_value.get_pipeline_config.return_value = PipelineConfig(project_id=pid)
            mgr._update_pipeline_components(pid, ComponentType.SOURCE)
            mgr._update_pipeline_components(pid, ComponentType.SINK)

        assert mgr.get_model_status().status == ModelStatus.READY
        mock_component_factory.create_processor.assert_not_called()

    def test_status_set_to_exception_message_when_processor_rebuild_fails(
        self, dispatcher, session_factory, mock_component_factory
    ):
        with patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls:
            batch_svc_cls.return_value.build.return_value = None
            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
        pid = uuid4()
        activate(mgr, Mock(), project_id=pid)

        mock_component_factory.create_processor.side_effect = RuntimeError("boom")

        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            pytest.raises(RuntimeError),
        ):
            svc_cls.return_value.get_pipeline_config.return_value = PipelineConfig(project_id=pid)
            mgr._update_pipeline_components(pid, ComponentType.PROCESSOR)

        assert mgr.is_model_loading() is False
        status = mgr.get_model_status()
        assert status.status == ModelStatus.ERROR
        assert status.error_type == ModelStatusErrorType.LOAD_FAILED
        assert status.error_message == "boom"

    def test_status_set_to_dinov3_error_when_processor_rebuild_hits_gated_weights(
        self, dispatcher, session_factory, mock_component_factory
    ):
        with patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls:
            batch_svc_cls.return_value.build.return_value = None
            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
        pid = uuid4()
        activate(mgr, Mock(), project_id=pid)

        error_message = (
            "User does not have access to the weights of the DinoV3 model.\n"
            "Please follow these steps:\n"
            "1. Request access on the HuggingFace website: "
            "https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m\n"
            "2. Set your HuggingFace credentials using one of these methods:\n"
            "   - Run: hf auth login\n"
            "   - Set environment variable: export HUGGINGFACE_HUB_TOKEN=your_token"
        )
        mock_component_factory.create_processor.side_effect = ValueError(error_message)

        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            pytest.raises(ValueError),
        ):
            svc_cls.return_value.get_pipeline_config.return_value = PipelineConfig(project_id=pid)
            mgr._update_pipeline_components(pid, ComponentType.PROCESSOR)

        status = mgr.get_model_status()
        assert status.status == ModelStatus.ERROR
        assert status.error_type == ModelStatusErrorType.ACCESS_REQUIRED
        assert status.error_message == _ACCESS_REQUIRED_MESSAGE

    def test_status_set_to_auth_error_when_processor_rebuild_hits_gated_repo(
        self, dispatcher, session_factory, mock_component_factory
    ):
        with patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls:
            batch_svc_cls.return_value.build.return_value = None
            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
        pid = uuid4()
        activate(mgr, Mock(), project_id=pid)

        auth_exc = OSError(
            "You are trying to access a gated repo. Make sure to have access to it at "
            "https://huggingface.co/facebook/sam3.1. Please log in."
        )
        mock_component_factory.create_processor.side_effect = auth_exc

        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            pytest.raises(OSError),
        ):
            svc_cls.return_value.get_pipeline_config.return_value = PipelineConfig(project_id=pid)
            mgr._update_pipeline_components(pid, ComponentType.PROCESSOR)

        status = mgr.get_model_status()
        assert status.status == ModelStatus.ERROR
        assert status.error_type == ModelStatusErrorType.AUTH_REQUIRED
        assert status.error_message == _AUTH_REQUIRED_MESSAGE

    def test_successful_rebuild_clears_previous_error(self, dispatcher, session_factory, mock_component_factory):
        with patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls:
            batch_svc_cls.return_value.build.return_value = None
            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
        pid = uuid4()
        activate(mgr, Mock(), project_id=pid)
        mgr._store_model_status(
            ModelStatusSchema(
                status=ModelStatus.ERROR,
                error_type=ModelStatusErrorType.LOAD_FAILED,
                error_message="old error",
            )
        )

        with patch("runtime.pipeline_manager.ProjectService") as svc_cls:
            svc_cls.return_value.get_pipeline_config.return_value = PipelineConfig(project_id=pid)
            mgr._update_pipeline_components(pid, ComponentType.PROCESSOR)

        status = mgr.get_model_status()
        assert status.status == ModelStatus.READY
        assert status.error_type is None
        assert status.error_message is None

    def test_status_is_cleared_when_the_processor_is_disposed(self, dispatcher, session_factory):
        """The status must not outlive the processor it describes."""
        with patch("runtime.pipeline_manager.FrameRepository"), patch("runtime.pipeline_manager.ReferenceBatchService"):
            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            activate(mgr, Mock(), project_id=uuid4())
            mgr._store_model_status(
                ModelStatusSchema(
                    status=ModelStatus.ERROR, error_type=ModelStatusErrorType.LOAD_FAILED, error_message="boom"
                )
            )

            mgr._teardown_pipeline()

            assert mgr.get_model_status().status is None
            assert mgr.get_model_status().error_message is None

    def test_error_status_does_not_leak_into_the_next_project(self, dispatcher, session_factory):
        with (
            patch("runtime.pipeline_manager.ProjectService"),
            patch("runtime.pipeline_manager.FrameRepository"),
            patch("runtime.pipeline_manager.ReferenceBatchService"),
        ):
            pid = uuid4()
            mgr = PipelineManager(dispatcher, session_factory, component_factory=Mock())
            activate(mgr, Mock(), project_id=pid)
            mgr._store_model_status(
                ModelStatusSchema(
                    status=ModelStatus.ERROR, error_type=ModelStatusErrorType.LOAD_FAILED, error_message="boom"
                )
            )

            mgr.on_config_change(ProjectDeactivationEvent(project_id=pid))

            assert mgr.get_model_status().status is None

    def test_reload_keeps_loading_status_while_tearing_down(
        self, dispatcher, session_factory, pipeline_cfg, mock_component_factory
    ):
        """The blocking UI overlay must not flicker between teardown and rebuild."""
        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.Pipeline") as pipeline_cls,
            patch("runtime.pipeline_manager.FrameBroadcaster"),
            patch("runtime.pipeline_manager.FrameRepository"),
            patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls,
            patch.object(PipelineManager, "_load_visualization_info", return_value=None),
        ):
            svc_cls.return_value.get_pipeline_config.return_value = pipeline_cfg
            batch_svc_cls.return_value.build.return_value = None
            pipeline_inst = pipeline_cls.return_value
            pipeline_inst.set_source.return_value = pipeline_inst
            pipeline_inst.set_processor.return_value = pipeline_inst
            pipeline_inst.set_sink.return_value = pipeline_inst

            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
            old_pipeline = activate(mgr, Mock(), project_id=pipeline_cfg.project_id)

            observed: list[ModelStatus] = []
            old_pipeline.stop.side_effect = lambda: observed.append(mgr.get_model_status().status)

            mgr.reload_pipeline(pipeline_cfg.project_id)

            assert observed == [ModelStatus.LOADING]

    def test_reload_pipeline_restarts_full_pipeline(
        self, dispatcher, session_factory, pipeline_cfg, mock_component_factory
    ):
        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.Pipeline") as pipeline_cls,
            patch("runtime.pipeline_manager.FrameBroadcaster"),
            patch("runtime.pipeline_manager.FrameRepository") as repo_cls,
            patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls,
            patch.object(PipelineManager, "_load_visualization_info", return_value=None),
        ):
            svc_cls.return_value.get_pipeline_config.return_value = pipeline_cfg
            batch_svc_cls.return_value.build.return_value = None
            repo_inst = repo_cls.return_value
            pipeline_inst = pipeline_cls.return_value
            pipeline_inst.set_source.return_value = pipeline_inst
            pipeline_inst.set_processor.return_value = pipeline_inst
            pipeline_inst.set_sink.return_value = pipeline_inst

            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
            old_pipeline = activate(mgr, Mock(), project_id=pipeline_cfg.project_id)

            mgr.reload_pipeline(pipeline_cfg.project_id)

            old_pipeline.stop.assert_called_once()
            pipeline_cls.assert_called_once()
            call_args = pipeline_cls.call_args.args
            assert call_args[0] == pipeline_cfg.project_id
            assert call_args[1] == repo_inst
            pipeline_inst.start.assert_called_once()
            assert mgr._state.pipeline == pipeline_inst
            assert mgr.get_model_status().status == ModelStatus.READY

    def test_reload_pipeline_raises_conflict_when_another_lifecycle_operation_runs(
        self, dispatcher, session_factory, mock_component_factory
    ):
        """The conflict check must be atomic: it is the lifecycle lock itself."""
        mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)

        holding = threading.Event()
        release = threading.Event()

        def hold_lifecycle_lock() -> None:
            with mgr._lifecycle_lock:
                holding.set()
                release.wait(timeout=5)

        holder = threading.Thread(target=hold_lifecycle_lock)
        holder.start()
        assert holding.wait(timeout=5)
        try:
            with (
                patch("runtime.pipeline_manager._RELOAD_LOCK_TIMEOUT_S", 0.05),
                pytest.raises(PipelineReloadInProgressError),
            ):
                mgr.reload_pipeline(uuid4())
        finally:
            release.set()
            holder.join(timeout=5)


class TestPipelineManagerBuildConcurrency:
    """The model build must never run while holding the state lock.

    ``get_output_slot()`` is called from the asyncio event loop when a WebRTC
    client connects; if a multi-minute export held ``_state_lock``, the whole
    event loop would stall.
    """

    def test_state_lock_is_free_while_the_processor_is_built(
        self, dispatcher, session_factory, pipeline_cfg, mock_component_factory
    ):
        with patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls:
            batch_svc_cls.return_value.build.return_value = None
            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)

        lock_free_during_build: list[bool] = []

        def probe_lock_from_another_thread() -> None:
            """``_state_lock`` is re-entrant, so it must be probed off the building thread."""
            acquired = mgr._state_lock.acquire(blocking=False)
            lock_free_during_build.append(acquired)
            if acquired:
                mgr._state_lock.release()

        def slow_create_processor(*args, **kwargs):
            probe = threading.Thread(target=probe_lock_from_another_thread)
            probe.start()
            probe.join(timeout=5)
            return Mock()

        mock_component_factory.create_processor.side_effect = slow_create_processor

        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.Pipeline") as pipeline_cls,
            patch("runtime.pipeline_manager.FrameRepository"),
            patch.object(PipelineManager, "_load_visualization_info", return_value=None),
        ):
            svc_cls.return_value.get_pipeline_config.return_value = pipeline_cfg
            pipeline_inst = pipeline_cls.return_value
            pipeline_inst.set_source.return_value = pipeline_inst
            pipeline_inst.set_processor.return_value = pipeline_inst
            pipeline_inst.set_sink.return_value = pipeline_inst

            mgr._build_and_start_pipeline(pipeline_cfg.project_id)

        assert lock_free_during_build == [True], "the state lock must not be held during the model build"

    def test_status_polling_is_not_blocked_by_a_running_build(
        self, dispatcher, session_factory, pipeline_cfg, mock_component_factory
    ):
        """``/model-status`` must stay responsive while a model is loading."""
        with patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls:
            batch_svc_cls.return_value.build.return_value = None
            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)

        observed: list[ModelStatus] = []

        def poll_status_from_another_thread(*args, **kwargs):
            probe = threading.Thread(target=lambda: observed.append(mgr.get_model_status().status))
            probe.start()
            probe.join(timeout=5)
            assert not probe.is_alive(), "get_model_status() blocked while the model was building"
            return Mock()

        mock_component_factory.create_processor.side_effect = poll_status_from_another_thread

        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.Pipeline") as pipeline_cls,
            patch("runtime.pipeline_manager.FrameRepository"),
            patch.object(PipelineManager, "_load_visualization_info", return_value=None),
        ):
            svc_cls.return_value.get_pipeline_config.return_value = pipeline_cfg
            pipeline_inst = pipeline_cls.return_value
            pipeline_inst.set_source.return_value = pipeline_inst
            pipeline_inst.set_processor.return_value = pipeline_inst
            pipeline_inst.set_sink.return_value = pipeline_inst

            mgr._build_and_start_pipeline(pipeline_cfg.project_id)

        assert observed == [ModelStatus.LOADING]

    def test_cancelled_build_is_discarded_and_stays_silent(
        self, dispatcher, session_factory, pipeline_cfg, mock_component_factory
    ):
        """A build cancelled by a teardown must not install itself nor publish a status."""
        with patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls:
            batch_svc_cls.return_value.build.return_value = None
            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)

        def teardown_midway(*args, **kwargs):
            # Simulate a project deactivation landing while the model is loading.
            mgr._teardown_pipeline()
            return Mock()

        mock_component_factory.create_processor.side_effect = teardown_midway

        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.Pipeline") as pipeline_cls,
            patch("runtime.pipeline_manager.FrameRepository"),
            patch.object(PipelineManager, "_load_visualization_info", return_value=None),
        ):
            svc_cls.return_value.get_pipeline_config.return_value = pipeline_cfg
            pipeline_inst = pipeline_cls.return_value
            pipeline_inst.set_source.return_value = pipeline_inst
            pipeline_inst.set_processor.return_value = pipeline_inst
            pipeline_inst.set_sink.return_value = pipeline_inst

            mgr._build_and_start_pipeline(pipeline_cfg.project_id)

        assert mgr._state is None
        pipeline_inst.start.assert_not_called()
        # The teardown owns the status; the discarded build must not overwrite it.
        assert mgr.get_model_status().status is None
        assert mgr._active_build is None

    def test_late_build_does_not_resurrect_a_torn_down_pipeline(
        self, dispatcher, session_factory, pipeline_cfg, mock_component_factory
    ):
        """Cancellation observed only after the pipeline object exists still discards it."""
        with patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls:
            batch_svc_cls.return_value.build.return_value = None
            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)

        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.Pipeline") as pipeline_cls,
            patch("runtime.pipeline_manager.FrameBroadcaster"),
            patch("runtime.pipeline_manager.FrameRepository"),
            patch.object(PipelineManager, "_load_visualization_info", return_value=None),
        ):
            svc_cls.return_value.get_pipeline_config.return_value = pipeline_cfg
            pipeline_inst = pipeline_cls.return_value
            pipeline_inst.set_source.return_value = pipeline_inst
            pipeline_inst.set_processor.return_value = pipeline_inst
            # The teardown lands after the processor was built and the pipeline assembled.
            pipeline_inst.set_sink.side_effect = lambda *a, **kw: (mgr._teardown_pipeline(), pipeline_inst)[1]

            mgr._build_and_start_pipeline(pipeline_cfg.project_id)

        assert mgr._state is None
        pipeline_inst.start.assert_not_called()
        pipeline_inst.stop.assert_called_once()

    def test_cancellation_before_start_never_transitions_the_pipeline_to_running(
        self, dispatcher, session_factory, pipeline_cfg, mock_component_factory
    ):
        """A teardown landing just before start() must not leave a running pipeline."""
        with patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls:
            batch_svc_cls.return_value.build.return_value = None
            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)

        def teardown_while_loading_visualization_info(*args, **kwargs):
            # Everything is built; only the install and start are left to do.
            mgr._teardown_pipeline()
            return None

        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.Pipeline") as pipeline_cls,
            patch("runtime.pipeline_manager.FrameBroadcaster"),
            patch("runtime.pipeline_manager.FrameRepository"),
            patch.object(
                PipelineManager, "_load_visualization_info", side_effect=teardown_while_loading_visualization_info
            ),
        ):
            svc_cls.return_value.get_pipeline_config.return_value = pipeline_cfg
            pipeline_inst = pipeline_cls.return_value
            pipeline_inst.set_source.return_value = pipeline_inst
            pipeline_inst.set_processor.return_value = pipeline_inst
            pipeline_inst.set_sink.return_value = pipeline_inst

            mgr._build_and_start_pipeline(pipeline_cfg.project_id)

        assert mgr._state is None
        pipeline_inst.start.assert_not_called()
        pipeline_inst.stop.assert_called_once()
        assert mgr.get_model_status().status is None

    def test_build_token_is_retired_after_a_successful_build(
        self, dispatcher, session_factory, pipeline_cfg, mock_component_factory
    ):
        with patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls:
            batch_svc_cls.return_value.build.return_value = None
            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)

        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.Pipeline") as pipeline_cls,
            patch("runtime.pipeline_manager.FrameBroadcaster"),
            patch("runtime.pipeline_manager.FrameRepository"),
            patch.object(PipelineManager, "_load_visualization_info", return_value=None),
        ):
            svc_cls.return_value.get_pipeline_config.return_value = pipeline_cfg
            pipeline_inst = pipeline_cls.return_value
            pipeline_inst.set_source.return_value = pipeline_inst
            pipeline_inst.set_processor.return_value = pipeline_inst
            pipeline_inst.set_sink.return_value = pipeline_inst

            mgr._build_and_start_pipeline(pipeline_cfg.project_id)

        assert mgr._active_build is None, "a finished build must not be cancellable by a later teardown"

    def test_build_token_is_retired_after_a_failed_processor_rebuild(
        self, dispatcher, session_factory, pipeline_cfg, mock_component_factory
    ):
        with patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls:
            batch_svc_cls.return_value.build.return_value = None
            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
        activate(mgr, Mock(), project_id=pipeline_cfg.project_id)

        mock_component_factory.create_processor.side_effect = RuntimeError("boom")

        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            pytest.raises(RuntimeError),
        ):
            svc_cls.return_value.get_pipeline_config.return_value = pipeline_cfg
            mgr._update_pipeline_components(pipeline_cfg.project_id, ComponentType.PROCESSOR)

        assert mgr._active_build is None

    def test_component_swap_is_skipped_when_pipeline_disappeared(
        self, dispatcher, session_factory, pipeline_cfg, mock_component_factory
    ):
        with patch("runtime.pipeline_manager.ReferenceBatchService") as batch_svc_cls:
            batch_svc_cls.return_value.build.return_value = None
            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
        running = activate(mgr, Mock(), project_id=pipeline_cfg.project_id)

        def teardown_midway(*args, **kwargs):
            mgr._teardown_pipeline()
            return Mock()

        mock_component_factory.create_processor.side_effect = teardown_midway

        with patch("runtime.pipeline_manager.ProjectService") as svc_cls:
            svc_cls.return_value.get_pipeline_config.return_value = pipeline_cfg
            mgr._update_pipeline_components(pipeline_cfg.project_id, ComponentType.PROCESSOR)

        assert mgr._state is None
        running.set_processor.assert_not_called()
        assert mgr.get_model_status().status is None
