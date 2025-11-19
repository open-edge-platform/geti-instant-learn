from queue import Queue
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from runtime.core.components.broadcaster import FrameBroadcaster
from runtime.core.components.pipeline import Pipeline
from runtime.core.components.processor import Processor
from runtime.core.components.sink import Sink
from runtime.core.components.source import Source


@pytest.fixture
def project_id():
    return uuid4()


@pytest.fixture
def mock_source():
    mock = Mock(spec=Source)
    mock.stop = Mock()
    mock.setup = Mock()
    return mock


@pytest.fixture
def mock_processor():
    mock = Mock(spec=Processor)
    mock.stop = Mock()
    mock.setup = Mock()
    return mock


@pytest.fixture
def mock_sink():
    mock = Mock(spec=Sink)
    mock.stop = Mock()
    mock.setup = Mock()
    return mock


@pytest.fixture
def mock_inbound_broadcaster():
    mock_broadcaster = Mock(spec=FrameBroadcaster)
    mock_broadcaster.register = Mock(return_value=Queue())
    mock_broadcaster.unregister = Mock()
    return mock_broadcaster


@pytest.fixture
def mock_outbound_broadcaster():
    mock_broadcaster = Mock(spec=FrameBroadcaster)
    mock_broadcaster.register = Mock(return_value=Queue())
    mock_broadcaster.unregister = Mock()
    return mock_broadcaster


class TestPipeline:
    def test_pipeline_initialization_with_no_components(self, project_id):
        """Test that Pipeline initializes with empty component dictionary."""
        pipeline = Pipeline(project_id=project_id)
        assert pipeline.project_id == project_id
        assert pipeline._components == {}
        pipeline.stop()

    def test_set_source_registers_component(self, project_id, mock_source, mock_inbound_broadcaster):
        """Test that set_source registers the source component."""
        pipeline = Pipeline(
            project_id=project_id,
            inbound_broadcaster=mock_inbound_broadcaster,
        )

        result = pipeline.set_source(mock_source)

        assert result is pipeline
        assert Source in pipeline._components
        assert pipeline._components[Source] == mock_source
        mock_source.setup.assert_called_once_with(mock_inbound_broadcaster)
        pipeline.stop()

    def test_set_processor_registers_component(
        self, project_id, mock_processor, mock_inbound_broadcaster, mock_outbound_broadcaster
    ):
        """Test that set_processor registers the processor component."""
        pipeline = Pipeline(
            project_id=project_id,
            inbound_broadcaster=mock_inbound_broadcaster,
            outbound_broadcaster=mock_outbound_broadcaster,
        )

        result = pipeline.set_processor(mock_processor)

        assert result is pipeline
        assert Processor in pipeline._components
        assert pipeline._components[Processor] == mock_processor
        mock_processor.setup.assert_called_once_with(mock_inbound_broadcaster, mock_outbound_broadcaster)
        pipeline.stop()

    def test_set_sink_registers_component(self, project_id, mock_sink, mock_outbound_broadcaster):
        """Test that set_sink registers the sink component."""
        pipeline = Pipeline(
            project_id=project_id,
            outbound_broadcaster=mock_outbound_broadcaster,
        )

        result = pipeline.set_sink(mock_sink)

        assert result is pipeline
        assert Sink in pipeline._components
        assert pipeline._components[Sink] == mock_sink
        mock_sink.setup.assert_called_once_with(mock_outbound_broadcaster)
        pipeline.stop()

    def test_pipeline_registers_and_unregisters_inbound_consumer(self, project_id, mock_inbound_broadcaster):
        """Test registering and unregistering inbound consumers."""
        pipeline = Pipeline(project_id=project_id, inbound_broadcaster=mock_inbound_broadcaster)

        consumer_queue = pipeline.register_inbound_consumer()
        mock_inbound_broadcaster.register.assert_called_once()
        assert isinstance(consumer_queue, Queue)

        pipeline.unregister_inbound_consumer(consumer_queue)
        mock_inbound_broadcaster.unregister.assert_called_once_with(consumer_queue)
        pipeline.stop()

    def test_pipeline_registers_and_unregisters_webrtc_consumer(self, project_id, mock_outbound_broadcaster):
        """Test registering and unregistering WebRTC consumers."""
        pipeline = Pipeline(project_id=project_id, outbound_broadcaster=mock_outbound_broadcaster)

        webrtc_queue = pipeline.register_webrtc()
        mock_outbound_broadcaster.register.assert_called_once()
        assert isinstance(webrtc_queue, Queue)

        pipeline.unregister_webrtc(webrtc_queue)
        mock_outbound_broadcaster.unregister.assert_called_once_with(queue=webrtc_queue)
        pipeline.stop()

    def test_pipeline_start_creates_threads_for_all_components(
        self, project_id, mock_source, mock_processor, mock_sink, mock_inbound_broadcaster, mock_outbound_broadcaster
    ):
        """Test that start() creates threads for all registered components."""
        pipeline = (
            Pipeline(
                project_id=project_id,
                inbound_broadcaster=mock_inbound_broadcaster,
                outbound_broadcaster=mock_outbound_broadcaster,
            )
            .set_source(mock_source)
            .set_processor(mock_processor)
            .set_sink(mock_sink)
        )

        with patch("runtime.core.components.pipeline.Thread") as mock_thread_class:
            mock_thread_instances = [Mock() for _ in range(3)]
            mock_thread_class.side_effect = mock_thread_instances

            pipeline.start()

            assert mock_thread_class.call_count == 3
            for mock_thread in mock_thread_instances:
                mock_thread.start.assert_called_once()

    def test_pipeline_stop_stops_components(
        self, project_id, mock_source, mock_processor, mock_sink, mock_inbound_broadcaster, mock_outbound_broadcaster
    ):
        """Test that stop() stops all components."""
        pipeline = (
            Pipeline(
                project_id=project_id,
                inbound_broadcaster=mock_inbound_broadcaster,
                outbound_broadcaster=mock_outbound_broadcaster,
            )
            .set_source(mock_source)
            .set_processor(mock_processor)
            .set_sink(mock_sink)
        )

        with patch("runtime.core.components.pipeline.Thread"):
            pipeline.start()

        pipeline.stop()

        mock_source.stop.assert_called_once()
        mock_processor.stop.assert_called_once()
        mock_sink.stop.assert_called_once()
