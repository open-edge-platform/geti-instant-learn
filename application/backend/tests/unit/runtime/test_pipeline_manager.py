#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import uuid
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, Mock, patch
from uuid import UUID, uuid4

import numpy as np
import pytest

from domain.db.models import PromptType
from domain.dispatcher import (
    ComponentConfigChangeEvent,
    ComponentType,
    ConfigChangeDispatcher,
    ProjectActivationEvent,
    ProjectDeactivationEvent,
)
from domain.services.schemas.pipeline import PipelineConfig
from domain.services.schemas.processor import MatcherConfig
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
            patch("runtime.pipeline_manager.FrameRepository") as repo_cls,
            patch.object(PipelineManager, "get_reference_batch", return_value=None),
            patch.object(PipelineManager, "_refresh_visualization_info", return_value=None),
        ):
            svc_inst = svc_cls.return_value
            svc_inst.get_active_pipeline_config.return_value = pipeline_cfg
            repo_inst = repo_cls.return_value

            # Configure the mock Pipeline to support method chaining
            pipeline_inst = pipeline_cls.return_value
            pipeline_inst.set_source.return_value = pipeline_inst
            pipeline_inst.set_processor.return_value = pipeline_inst
            pipeline_inst.set_sink.return_value = pipeline_inst

            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
            mgr.start()

            svc_inst.get_active_pipeline_config.assert_called_once()
            mock_component_factory.create_source.assert_called_once_with(pipeline_cfg.project_id)
            mock_component_factory.create_processor.assert_called_once_with(
                pipeline_cfg.project_id, None, status_reporter=ANY
            )
            mock_component_factory.create_sink.assert_called_once_with(pipeline_cfg.project_id)

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
        ):
            svc_inst = svc_cls.return_value
            svc_inst.get_active_pipeline_config.return_value = None

            mgr = PipelineManager(dispatcher, session_factory)
            mgr.start()

            svc_inst.get_active_pipeline_config.assert_called_once()
            pipeline_cls.assert_not_called()
            assert mgr._pipeline is None
            assert dispatcher._listeners == [mgr.on_config_change]

    def test_on_activation_event_starts_new_pipeline(
        self, dispatcher, session_factory, pipeline_cfg, mock_component_factory
    ):
        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.Pipeline") as pipeline_cls,
            patch("runtime.pipeline_manager.FrameBroadcaster"),
            patch("runtime.pipeline_manager.FrameRepository") as repo_cls,
            patch.object(PipelineManager, "get_reference_batch", return_value=None),
            patch.object(PipelineManager, "_refresh_visualization_info", return_value=None),
        ):
            pid = pipeline_cfg.project_id
            svc_cls.return_value.get_pipeline_config.return_value = pipeline_cfg
            repo_inst = repo_cls.return_value

            # Configure the mock Pipeline to support method chaining
            pipeline_inst = pipeline_cls.return_value
            pipeline_inst.set_source.return_value = pipeline_inst
            pipeline_inst.set_processor.return_value = pipeline_inst
            pipeline_inst.set_sink.return_value = pipeline_inst

            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
            ev = ProjectActivationEvent(project_id=pid)
            mgr.on_config_change(ev)

            mock_component_factory.create_source.assert_called_once_with(pid)
            mock_component_factory.create_processor.assert_called_once_with(pid, None, status_reporter=ANY)
            mock_component_factory.create_sink.assert_called_once_with(pid)

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

    def test_on_activation_replaces_existing_pipeline(
        self, dispatcher, session_factory, pipeline_cfg, mock_component_factory
    ):
        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.Pipeline") as pipeline_cls,
            patch("runtime.pipeline_manager.FrameBroadcaster"),
            patch("runtime.pipeline_manager.FrameRepository"),
            patch.object(PipelineManager, "get_reference_batch", return_value=None),
            patch.object(PipelineManager, "_refresh_visualization_info", return_value=None),
        ):
            # Existing pipeline
            old_pipeline = Mock()
            pid_new = pipeline_cfg.project_id
            svc_cls.return_value.get_pipeline_config.return_value = pipeline_cfg

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
            mock_component_factory.create_processor.assert_called_once_with(pid_new, None, status_reporter=ANY)
            mock_component_factory.create_sink.assert_called_once_with(pid_new)

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
        mgr = PipelineManager(dispatcher, session_factory)
        with pytest.raises(PipelineNotActiveError):
            mgr.get_visualization_info(uuid4())

    def test_get_visualization_info_raises_when_project_mismatched(self, dispatcher, session_factory):
        mgr = PipelineManager(dispatcher, session_factory)
        running = Mock()
        running.project_id = uuid4()
        mgr._pipeline = running

        with pytest.raises(PipelineProjectMismatchError):
            mgr.get_visualization_info(uuid4())

    def test_get_visualization_info_returns_cached_value(self, dispatcher, session_factory):
        mgr = PipelineManager(dispatcher, session_factory)
        pid = uuid4()
        running = Mock()
        running.project_id = pid
        mgr._pipeline = running

        cached = Mock()
        mgr._visualization_info = cached

        assert mgr.get_visualization_info(pid) is cached

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
            component_id = uuid4()
            running = Mock()
            running.project_id = pid

            mgr = PipelineManager(dispatcher, session_factory, component_factory=mock_component_factory)
            mgr._pipeline = running

            ev = ComponentConfigChangeEvent(
                project_id=pid, component_type=ComponentType.SOURCE, component_id=component_id
            )
            mgr.on_config_change(ev)

            mock_component_factory.create_source.assert_called_once_with(pid)
            running.set_source.assert_called_once()

    def test_on_component_update_ignores_mismatch(self, dispatcher, session_factory):
        with patch("runtime.pipeline_manager.Pipeline"):
            pid_running = uuid4()
            pid_event = uuid4()
            component_id = uuid4()
            running = Mock()
            running.project_id = pid_running

            mgr = PipelineManager(dispatcher, session_factory)
            mgr._pipeline = running

            ev = ComponentConfigChangeEvent(
                project_id=pid_event, component_type=ComponentType.SOURCE, component_id=component_id
            )
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

    def test_get_reference_batch_text_prompts_returns_none(self, dispatcher, session_factory) -> None:
        mgr = PipelineManager(dispatcher, session_factory)
        assert mgr.get_reference_batch(uuid4(), PromptType.TEXT) is None

    def test_get_reference_batch_for_visual_prompts_returns_batch_and_mapping(
        self, dispatcher, session_factory
    ) -> None:
        mgr = PipelineManager(dispatcher, session_factory)
        project_id = uuid4()
        frame_id = uuid4()
        label_id = uuid4()

        annotation_db = SimpleNamespace(
            id=uuid.uuid4(),
            config={
                "type": "polygon",
                "points": [{"x": 0.1, "y": 0.1}, {"x": 0.5, "y": 0.1}, {"x": 0.5, "y": 0.5}, {"x": 0.1, "y": 0.5}],
            },
            label_id=label_id,
        )
        visual_prompt = SimpleNamespace(id=uuid4(), frame_id=frame_id, annotations=[annotation_db])

        frame_bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        fake_sample = MagicMock(name="Sample")
        fake_batch = MagicMock(name="Batch")
        fake_batch.samples = [fake_sample]

        with (
            patch("runtime.pipeline_manager.PromptRepository") as prompt_repo_cls,
            patch("runtime.pipeline_manager.LabelService") as label_svc_cls,
            patch.object(mgr._frame_repository, "read_frame", return_value=frame_bgr) as read_frame,
            patch("runtime.pipeline_manager.cv2.cvtColor", return_value=np.zeros((64, 64, 3), dtype=np.uint8)),
            patch("runtime.pipeline_manager.visual_prompt_to_sample", return_value=fake_sample),
            patch("runtime.pipeline_manager.Batch.collate", return_value=fake_batch),
        ):
            prompt_repo_cls.return_value.list_all_by_project.return_value = [visual_prompt]
            label_svc_cls.return_value.build_category_mappings.return_value = SimpleNamespace(
                label_to_category_id={label_id: 0},
                category_id_to_label_id={0: str(label_id)},
            )

            result = mgr.get_reference_batch(project_id, PromptType.VISUAL)

        assert result is not None
        batch, category_id_to_label_id = result
        assert batch is fake_batch
        assert category_id_to_label_id == {0: str(label_id)}

        prompt_repo_cls.return_value.list_all_by_project.assert_called_once_with(
            project_id=project_id, prompt_type=PromptType.VISUAL
        )
        read_frame.assert_called_once_with(project_id, frame_id)

    def test_get_reference_batch_category_mapping_sorted_by_label_id_string(self, dispatcher, session_factory) -> None:
        mgr = PipelineManager(dispatcher, session_factory)
        project_id = uuid4()
        frame_id = uuid4()

        label_id_a = UUID("00000000-0000-0000-0000-00000000000a")
        label_id_b = UUID("00000000-0000-0000-0000-00000000000b")

        ann_1 = SimpleNamespace(id=uuid4(), config={"type": "polygon", "points": []}, label_id=label_id_b)
        ann_2 = SimpleNamespace(id=uuid4(), config={"type": "polygon", "points": []}, label_id=label_id_a)
        visual_prompt = SimpleNamespace(id=uuid4(), frame_id=frame_id, annotations=[ann_1, ann_2])

        frame_bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        fake_sample = MagicMock(name="Sample")
        fake_batch = MagicMock(name="Batch")
        fake_batch.samples = [fake_sample]

        with (
            patch("runtime.pipeline_manager.PromptRepository") as prompt_repo_cls,
            patch("runtime.pipeline_manager.LabelService") as label_svc_cls,
            patch.object(mgr._frame_repository, "read_frame", return_value=frame_bgr),
            patch("runtime.pipeline_manager.cv2.cvtColor", return_value=np.zeros((64, 64, 3), dtype=np.uint8)),
            patch("runtime.pipeline_manager.visual_prompt_to_sample", return_value=fake_sample),
            patch("runtime.pipeline_manager.Batch.collate", return_value=fake_batch),
        ):
            prompt_repo_cls.return_value.list_all_by_project.return_value = [visual_prompt]
            label_svc_cls.return_value.build_category_mappings.return_value = SimpleNamespace(
                label_to_category_id={label_id_a: 0, label_id_b: 1},
                category_id_to_label_id={0: str(label_id_a), 1: str(label_id_b)},
            )

            result = mgr.get_reference_batch(project_id, PromptType.VISUAL)

        assert result is not None
        _, category_id_to_label_id = result
        assert category_id_to_label_id == {0: str(label_id_a), 1: str(label_id_b)}

    def test_get_reference_batch_visual_prompts_empty_returns_none(self, dispatcher, session_factory) -> None:
        mgr = PipelineManager(dispatcher, session_factory)
        project_id = uuid4()

        with (
            patch("runtime.pipeline_manager.PromptRepository") as prompt_repo_cls,
            patch("runtime.pipeline_manager.LabelService"),
        ):
            prompt_repo_cls.return_value.list_all_by_project.return_value = []

            result = mgr.get_reference_batch(project_id, PromptType.VISUAL)

        assert result is None

    def test_get_reference_batch_visual_prompt_frame_not_found_returns_none(self, dispatcher, session_factory) -> None:
        mgr = PipelineManager(dispatcher, session_factory)
        project_id = uuid4()
        frame_id = uuid4()

        visual_prompt = SimpleNamespace(id=uuid4(), frame_id=frame_id, annotations=[])

        with (
            patch("runtime.pipeline_manager.PromptRepository") as prompt_repo_cls,
            patch("runtime.pipeline_manager.LabelService") as label_svc_cls,
            patch.object(mgr._frame_repository, "read_frame", return_value=None),
        ):
            prompt_repo_cls.return_value.list_all_by_project.return_value = [visual_prompt]
            label_svc_cls.return_value.build_category_mappings.return_value = SimpleNamespace(
                label_to_category_id={}, category_id_to_label_id={}
            )

            result = mgr.get_reference_batch(project_id, PromptType.VISUAL)

        assert result is None

    def test_get_reference_batch_visual_prompt_mapper_error_handled_returns_none(
        self, dispatcher, session_factory
    ) -> None:
        mgr = PipelineManager(dispatcher, session_factory)
        project_id = uuid4()
        frame_id = uuid4()

        visual_prompt = SimpleNamespace(id=uuid4(), frame_id=frame_id, annotations=[])

        frame_bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        with (
            patch("runtime.pipeline_manager.PromptRepository") as prompt_repo_cls,
            patch("runtime.pipeline_manager.LabelService") as label_svc_cls,
            patch.object(mgr._frame_repository, "read_frame", return_value=frame_bgr),
            patch("runtime.pipeline_manager.cv2.cvtColor", return_value=np.zeros((64, 64, 3), dtype=np.uint8)),
            patch("runtime.pipeline_manager.visual_prompt_to_sample", side_effect=Exception("Mapper error")),
        ):
            prompt_repo_cls.return_value.list_all_by_project.return_value = [visual_prompt]
            label_svc_cls.return_value.build_category_mappings.return_value = SimpleNamespace(
                label_to_category_id={}, category_id_to_label_id={}
            )

            result = mgr.get_reference_batch(project_id, PromptType.VISUAL)

        assert result is None


class TestPipelineManagerStatus:
    """Tests for model status publishing behaviour."""

    def _make_mgr(self, dispatcher, session_factory, component_factory=None):
        mgr = PipelineManager(dispatcher, session_factory, component_factory=component_factory)
        return mgr

    def test_stop_always_publishes_idle(self, dispatcher, session_factory):
        """stop() must publish IDLE even if the pipeline's stop() raises."""
        mgr = self._make_mgr(dispatcher, session_factory)
        failing_pipeline = Mock()
        failing_pipeline.stop.side_effect = RuntimeError("boom")
        mgr._pipeline = failing_pipeline

        with pytest.raises(RuntimeError):
            mgr.stop()

        assert mgr.get_status().state.value == "idle"

    def test_stop_publishes_idle_when_no_pipeline(self, dispatcher, session_factory):
        mgr = self._make_mgr(dispatcher, session_factory)
        mgr._pipeline = None
        mgr.stop()
        assert mgr.get_status().state.value == "idle"

    def test_prepare_processor_publishes_loading_reference_batch_then_loading_model(
        self, dispatcher, session_factory, pipeline_cfg, mock_component_factory
    ):
        """_prepare_processor emits LOADING_REFERENCE_BATCH then LOADING_MODEL when cfg.processor is set."""
        from unittest.mock import MagicMock

        from instantlearn.data.base.batch import Batch

        fake_batch = MagicMock(spec=Batch)
        fake_batch.samples = [MagicMock()]

        # Give cfg a processor config so LOADING_MODEL is emitted.
        pipeline_cfg_with_processor = PipelineConfig(
            project_id=pipeline_cfg.project_id,
            reader=None,
            processor=MatcherConfig(),
            writer=None,
        )

        with (
            patch.object(PipelineManager, "get_reference_batch", return_value=(fake_batch, {})),
        ):
            mgr = self._make_mgr(dispatcher, session_factory, component_factory=mock_component_factory)
            reporter = Mock()
            published_states = []
            original_publish = mgr._publish_status

            def track(status):
                published_states.append(status.state.value)
                original_publish(status)

            mgr._publish_status = track

            mgr._prepare_processor(pipeline_cfg_with_processor, reporter)

        assert published_states == ["loading_reference_batch", "loading_model"]

    def test_prepare_processor_skips_loading_model_when_processor_config_is_none(
        self, dispatcher, session_factory, pipeline_cfg, mock_component_factory
    ):
        """_prepare_processor skips LOADING_MODEL when cfg.processor is None (passthrough)."""
        with patch.object(PipelineManager, "get_reference_batch", return_value=None):
            mgr = self._make_mgr(dispatcher, session_factory, component_factory=mock_component_factory)
            reporter = Mock()
            published_states = []
            original_publish = mgr._publish_status

            def track(status):
                published_states.append(status.state.value)
                original_publish(status)

            mgr._publish_status = track

            mgr._prepare_processor(pipeline_cfg, reporter)

        assert published_states == ["loading_reference_batch"]

    def test_prepare_processor_skips_loading_model_when_reference_batch_is_none(
        self, dispatcher, session_factory, mock_component_factory
    ):
        """_prepare_processor skips LOADING_MODEL when reference batch is None (no prompts)."""
        cfg = PipelineConfig(
            project_id=uuid4(),
            reader=None,
            processor=MatcherConfig(),
            writer=None,
        )

        with patch.object(PipelineManager, "get_reference_batch", return_value=None):
            mgr = self._make_mgr(dispatcher, session_factory, component_factory=mock_component_factory)
            reporter = Mock()
            published_states = []
            original_publish = mgr._publish_status

            def track(status):
                published_states.append(status.state.value)
                original_publish(status)

            mgr._publish_status = track

            mgr._prepare_processor(cfg, reporter)

        assert published_states == ["loading_reference_batch"]

    def test_start_publishes_idle_when_no_active_project(self, dispatcher, session_factory):
        with (
            patch("runtime.pipeline_manager.ProjectService") as svc_cls,
            patch("runtime.pipeline_manager.FrameRepository"),
        ):
            svc_cls.return_value.get_active_pipeline_config.return_value = None
            mgr = self._make_mgr(dispatcher, session_factory)
            mgr.start()

        assert mgr.get_status().state.value == "idle"

    def test_subscribe_and_get_status(self, dispatcher, session_factory):
        mgr = self._make_mgr(dispatcher, session_factory)
        status = mgr.get_status()
        assert status.state.value == "idle"

    def test_on_processor_update_publishes_error_on_failure(self, dispatcher, session_factory, mock_component_factory):
        """When _prepare_processor fails, an ERROR status is published."""
        with patch("runtime.pipeline_manager.ProjectService") as svc_cls:
            pid = uuid4()
            cfg = PipelineConfig(project_id=pid, reader=None, processor=MatcherConfig(), writer=None)
            svc_cls.return_value.get_pipeline_config.return_value = cfg

            running = Mock()
            running.project_id = pid

            mgr = self._make_mgr(dispatcher, session_factory, component_factory=mock_component_factory)
            mgr._pipeline = running

            with patch.object(PipelineManager, "_prepare_processor", side_effect=RuntimeError("download failed")):
                ev = ComponentConfigChangeEvent(
                    project_id=pid, component_type=ComponentType.PROCESSOR, component_id=uuid4()
                )
                with pytest.raises(RuntimeError):
                    mgr.on_config_change(ev)

            assert mgr.get_status().state.value == "error"
