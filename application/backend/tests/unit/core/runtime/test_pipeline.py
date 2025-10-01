from queue import Queue
from typing import Any
from unittest.mock import Mock, patch

import pytest

from core.components.broadcaster import FrameBroadcaster
from core.components.factories.components import ComponentFactory
from core.components.processor import Processor
from core.components.sink import Sink
from core.components.source import Source
from core.runtime.pipeline import Pipeline
from core.runtime.schemas.pipeline import PipelineConfig


class MockComponentFactory(ComponentFactory):
    """Mock implementation of ComponentFactory for testing."""

    def __init__(self):
        self.created_sources = []
        self.created_processors = []
        self.created_sinks = []

    def create_source(self, in_queue: Queue, source_config: Any) -> Mock:
        mock_source = Mock(spec=Source)
        mock_source.stop = Mock()
        self.created_sources.append((mock_source, in_queue, source_config))
        return mock_source

    def create_processor(self, in_queue: Queue, broadcaster: FrameBroadcaster, model_config: Any) -> Mock:
        mock_processor = Mock(spec=Processor)
        mock_processor.stop = Mock()
        self.created_processors.append((mock_processor, in_queue, broadcaster, model_config))
        return mock_processor

    def create_sink(self, broadcaster: FrameBroadcaster, sink_config: Any) -> Mock:
        mock_sink = Mock(spec=Sink)
        mock_sink.stop = Mock()
        self.created_sinks.append((mock_sink, broadcaster, sink_config))
        return mock_sink


@pytest.fixture
def mock_config():
    mock_config = Mock(spec=PipelineConfig)
    mock_config.project_id = "test-project-123"
    mock_config.reader = {"type": "test_source"}
    mock_config.processor = {"type": "test_pipeline"}
    mock_config.writer = {"type": "test_sink"}

    mock_config.model_copy.return_value = mock_config

    return Mock()


@pytest.fixture
def mock_factory():
    return MockComponentFactory()


class TestPipeline:
    def test_pipeline_initialization_creates_components(self, mock_config, mock_factory):
        pipeline = Pipeline(mock_config, component_factory=mock_factory)
        pipeline.stop()
        assert len(mock_factory.created_sources) == 1

        assert len(mock_factory.created_processors) == 1
        assert len(mock_factory.created_sinks) == 1

        _, _, source_config = mock_factory.created_sources[0]

        assert source_config == mock_config.reader

    def test_pipeline_start_creates_threads_for_all_components(self, mock_config, mock_factory):
        pipelines = Pipeline(mock_config, component_factory=mock_factory)

        with patch("core.runtime.pipeline.Thread") as mock_thread_class:
            mock_thread_instances = [Mock() for _ in range(3)]
            mock_thread_class.side_effect = mock_thread_instances

            pipelines.start()

            assert mock_thread_class.call_count == 3
            for mock_thread in mock_thread_instances:
                mock_thread.start.assert_called_once()

    def test_pipeline_stop_stops_components_in_correct_order(self, mock_config, mock_factory):
        pipeline = Pipeline(mock_config, component_factory=mock_factory)

        with patch("core.runtime.pipeline.Thread"):
            pipeline.start()

        pipeline.stop()

        source_mock = mock_factory.created_sources[0][0]
        pipeline_mock = mock_factory.created_processors[0][0]
        sink_mock = mock_factory.created_sinks[0][0]

        source_mock.stop.assert_called_once()
        pipeline_mock.stop.assert_called_once()
        sink_mock.stop.assert_called_once()

    def test_update_config_restarts_source_when_config_changes(self, mock_config, mock_factory):
        pipeline = Pipeline(mock_config, component_factory=mock_factory)

        new_config = Mock(spec=PipelineConfig)
        new_config.project_id = mock_config.project_id
        new_config.reader = {"type": "new_source"}  # Different config
        new_config.processor = mock_config.processor  # Same
        new_config.writer = mock_config.writer  # Same

        with patch("core.runtime.pipeline.Thread") as mock_thread:
            pipeline.start()
            mock_thread.reset_mock()

            pipeline.update_config(new_config)

            # Should create one new source, adding 1 to a created source when the job started.
            assert len(mock_factory.created_sources) == 2

            # Should not create new pipeline or sink:
            assert len(mock_factory.created_processors) == 1
            assert len(mock_factory.created_sinks) == 1

            # Should create one new thread for the restarted source
            mock_thread.assert_called_once()

    def test_update_config_no_restart_when_config_unchanged(self, mock_config, mock_factory):
        pipeline = Pipeline(mock_config, component_factory=mock_factory)

        # Same config
        same_config = Mock(spec=PipelineConfig)
        same_config.reader = mock_config.reader
        same_config.processor = mock_config.processor
        same_config.writer = mock_config.writer

        with patch("core.runtime.pipeline.Thread"):
            pipeline.start()

            original_source = mock_factory.created_sources[0][0]
            original_pipeline = mock_factory.created_processors[0][0]
            original_sink = mock_factory.created_sinks[0][0]

            # Clear creation tracking
            mock_factory.created_sources.clear()
            mock_factory.created_processors.clear()
            mock_factory.created_sinks.clear()

            pipeline.update_config(same_config)

            # no new components should be created
            assert len(mock_factory.created_sources) == 0
            assert len(mock_factory.created_processors) == 0
            assert len(mock_factory.created_sinks) == 0

            # Original components should not be stopped
            original_source.stop.assert_not_called()
            original_pipeline.stop.assert_not_called()
            original_sink.stop.assert_not_called()

    def test_update_config_restarts_all_components(self, mock_config, mock_factory):
        pipeline = Pipeline(mock_config, component_factory=mock_factory)

        new_config = Mock(spec=PipelineConfig)
        new_config.project_id = mock_config.project_id
        new_config.reader = {"type": "new_source"}  # Different
        new_config.processor = {"type": "new_pipeline"}  # Different
        new_config.writer = {"type": "new_sink"}  # Same

        with patch("core.runtime.pipeline.Thread"):
            pipeline.start()

            original_source = mock_factory.created_sources[0][0]
            original_pipeline = mock_factory.created_processors[0][0]
            original_sink = mock_factory.created_sinks[0][0]

            # Clear creation tracking
            mock_factory.created_sources.clear()
            mock_factory.created_processors.clear()
            mock_factory.created_sinks.clear()

            pipeline.update_config(new_config)

            assert len(mock_factory.created_sources) == 1
            assert len(mock_factory.created_processors) == 1
            assert len(mock_factory.created_sinks) == 1

            # Original source and pipeline should be stopped
            original_source.stop.assert_called_once()
            original_pipeline.stop.assert_called_once()
            original_sink.stop.assert_called_once()
