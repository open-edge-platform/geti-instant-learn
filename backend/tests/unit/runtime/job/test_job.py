from queue import Queue
from typing import Any
from unittest.mock import Mock, patch

import pytest

from runtime.core.components.broadcaster import FrameBroadcaster
from runtime.core.components.factories.components import ComponentFactory
from runtime.core.components.pipeline import PipelineRunner
from runtime.core.components.sink import Sink
from runtime.core.components.source import Source
from runtime.job.job import Job
from runtime.job.schemas.project import ProjectConfig


class MockComponentFactory(ComponentFactory):
    """Mock implementation of ComponentFactory for testing."""

    def __init__(self):
        self.created_sources = []
        self.created_pipeline_runners = []
        self.created_sinks = []

    def create_source(self, in_queue: Queue, source_config: Any) -> Mock:
        mock_source = Mock(spec=Source)
        mock_source.stop = Mock()
        self.created_sources.append((mock_source, in_queue, source_config))
        return mock_source

    def create_pipeline(self, in_queue: Queue, broadcaster: FrameBroadcaster, pipeline_config: Any) -> Mock:
        mock_runner = Mock(spec=PipelineRunner)
        mock_runner.stop = Mock()
        self.created_pipeline_runners.append((mock_runner, in_queue, broadcaster, pipeline_config))
        return mock_runner

    def create_sink(self, broadcaster: FrameBroadcaster, sink_config: Any) -> Mock:
        mock_sink = Mock(spec=Sink)
        mock_sink.stop = Mock()
        self.created_sinks.append((mock_sink, broadcaster, sink_config))
        return mock_sink


@pytest.fixture
def mock_config():
    mock_config = Mock(spec=ProjectConfig)
    mock_config.project_id = "test-project-123"
    mock_config.reader = {"type": "test_source"}
    mock_config.processor = {"type": "test_pipeline"}
    mock_config.writer = {"type": "test_sink"}

    mock_config.model_copy.return_value = mock_config

    return Mock()


@pytest.fixture
def mock_factory():
    return MockComponentFactory()


class TestJob:
    def test_job_initialization_creates_components(self, mock_config, mock_factory):
        job = Job(mock_config, component_factory=mock_factory)
        job.stop()
        assert len(mock_factory.created_sources) == 1

        assert len(mock_factory.created_pipeline_runners) == 1
        assert len(mock_factory.created_sinks) == 1

        _, _, source_config = mock_factory.created_sources[0]

        assert source_config == mock_config.reader

    def test_job_start_creates_threads_for_all_components(self, mock_config, mock_factory):
        job = Job(mock_config, component_factory=mock_factory)

        with patch("runtime.job.job.Thread") as mock_thread_class:
            mock_thread_instances = [Mock() for _ in range(3)]
            mock_thread_class.side_effect = mock_thread_instances

            job.start()

            assert mock_thread_class.call_count == 3
            for mock_thread in mock_thread_instances:
                mock_thread.start.assert_called_once()

    def test_job_stop_stops_components_in_correct_order(self, mock_config, mock_factory):
        job = Job(mock_config, component_factory=mock_factory)

        with patch("runtime.job.job.Thread"):
            job.start()

        job.stop()

        source_mock = mock_factory.created_sources[0][0]
        pipeline_mock = mock_factory.created_pipeline_runners[0][0]
        sink_mock = mock_factory.created_sinks[0][0]

        source_mock.stop.assert_called_once()
        pipeline_mock.stop.assert_called_once()
        sink_mock.stop.assert_called_once()

    def test_update_config_restarts_source_when_config_changes(self, mock_config, mock_factory):
        job = Job(mock_config, component_factory=mock_factory)

        new_config = Mock(spec=ProjectConfig)
        new_config.project_id = mock_config.project_id
        new_config.reader = {"type": "new_source"}  # Different config
        new_config.processor = mock_config.processor  # Same
        new_config.writer = mock_config.writer  # Same

        with patch("runtime.job.job.Thread") as mock_thread:
            job.start()
            mock_thread.reset_mock()

            job.update_config(new_config)

            # Should create one new source, adding 1 to a created source when the job started.
            assert len(mock_factory.created_sources) == 2

            # Should not create new pipeline or sink:
            assert len(mock_factory.created_pipeline_runners) == 1
            assert len(mock_factory.created_sinks) == 1

            # Should create one new thread for the restarted source
            mock_thread.assert_called_once()

    def test_update_config_no_restart_when_config_unchanged(self, mock_config, mock_factory):
        job = Job(mock_config, component_factory=mock_factory)

        # Same config
        same_config = Mock(spec=ProjectConfig)
        same_config.reader = mock_config.reader
        same_config.processor = mock_config.processor
        same_config.writer = mock_config.writer

        with patch("runtime.job.job.Thread"):
            job.start()

            original_source = mock_factory.created_sources[0][0]
            original_pipeline = mock_factory.created_pipeline_runners[0][0]
            original_sink = mock_factory.created_sinks[0][0]

            # Clear creation tracking
            mock_factory.created_sources.clear()
            mock_factory.created_pipeline_runners.clear()
            mock_factory.created_sinks.clear()

            job.update_config(same_config)

            # no new components should be created
            assert len(mock_factory.created_sources) == 0
            assert len(mock_factory.created_pipeline_runners) == 0
            assert len(mock_factory.created_sinks) == 0

            # Original components should not be stopped
            original_source.stop.assert_not_called()
            original_pipeline.stop.assert_not_called()
            original_sink.stop.assert_not_called()

    def test_update_config_restarts_all_components(self, mock_config, mock_factory):
        job = Job(mock_config, component_factory=mock_factory)

        new_config = Mock(spec=ProjectConfig)
        new_config.project_id = mock_config.project_id
        new_config.reader = {"type": "new_source"}  # Different
        new_config.processor = {"type": "new_pipeline"}  # Different
        new_config.writer = {"type": "new_sink"}  # Same

        with patch("runtime.job.job.Thread"):
            job.start()

            original_source = mock_factory.created_sources[0][0]
            original_pipeline = mock_factory.created_pipeline_runners[0][0]
            original_sink = mock_factory.created_sinks[0][0]

            # Clear creation tracking
            mock_factory.created_sources.clear()
            mock_factory.created_pipeline_runners.clear()
            mock_factory.created_sinks.clear()

            job.update_config(new_config)

            assert len(mock_factory.created_sources) == 1
            assert len(mock_factory.created_pipeline_runners) == 1
            assert len(mock_factory.created_sinks) == 1

            # Original source and pipeline should be stopped
            original_source.stop.assert_called_once()
            original_pipeline.stop.assert_called_once()
            original_sink.stop.assert_called_once()
