from queue import Queue
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from core.components.processor import Processor
from core.components.sink import Sink
from core.components.source import Source
from core.runtime.pipeline import Pipeline


def create_mock_component(component_class):
    mock = Mock()
    mock.stop = Mock()
    mock.__class__ = component_class
    return mock


@pytest.fixture
def mock_source():
    return create_mock_component(Source)


@pytest.fixture
def mock_processor():
    return create_mock_component(Processor)


@pytest.fixture
def mock_sink():
    return create_mock_component(Sink)


@pytest.fixture
def project_id():
    return uuid4()


@pytest.fixture
def mock_inbound_broadcaster():
    """Mock inbound broadcaster that tracks register/unregister calls."""
    mock_broadcaster = Mock()
    mock_broadcaster.register = Mock(return_value=Queue())
    mock_broadcaster.unregister = Mock()
    return mock_broadcaster


@pytest.fixture
def mock_outbound_broadcaster():
    """Mock outbound broadcaster that tracks register/unregister calls."""
    mock_broadcaster = Mock()
    mock_broadcaster.register = Mock(return_value=Queue())
    mock_broadcaster.unregister = Mock()
    return mock_broadcaster


class TestPipeline:
    def test_pipeline_registers_and_unregisters_inbound_consumer(
        self, project_id, mock_source, mock_processor, mock_sink, mock_inbound_broadcaster
    ):
        """Test inbound consumer registration delegates to broadcaster."""
        pipeline = Pipeline(
            project_id=project_id,
            source=mock_source,
            processor=mock_processor,
            sink=mock_sink,
            inbound_broadcaster=mock_inbound_broadcaster,
        )

        consumer_queue = pipeline.register_inbound_consumer()
        mock_inbound_broadcaster.register.assert_called_once()
        assert isinstance(consumer_queue, Queue)

        pipeline.unregister_inbound_consumer(consumer_queue)
        mock_inbound_broadcaster.unregister.assert_called_once_with(consumer_queue)
        pipeline.stop()

    def test_pipeline_registers_and_unregisters_webrtc_consumer(
        self, project_id, mock_source, mock_processor, mock_sink, mock_outbound_broadcaster
    ):
        """Test WebRTC consumer registration delegates to broadcaster."""
        pipeline = Pipeline(
            project_id=project_id,
            source=mock_source,
            processor=mock_processor,
            sink=mock_sink,
            outbound_broadcaster=mock_outbound_broadcaster,
        )

        webrtc_queue = pipeline.register_webrtc()
        mock_outbound_broadcaster.register.assert_called_once()
        assert isinstance(webrtc_queue, Queue)

        pipeline.unregister_webrtc(webrtc_queue)
        mock_outbound_broadcaster.unregister.assert_called_once_with(queue=webrtc_queue)
        pipeline.stop()

    def test_pipeline_start_creates_threads_for_all_components(
        self, project_id, mock_source, mock_processor, mock_sink
    ):
        """Test Pipeline starts all components in threads."""
        pipeline = Pipeline(
            project_id=project_id,
            source=mock_source,
            processor=mock_processor,
            sink=mock_sink,
        )

        with patch("core.runtime.pipeline.Thread") as mock_thread_class:
            mock_thread_instances = [Mock() for _ in range(3)]
            mock_thread_class.side_effect = mock_thread_instances

            pipeline.start()

            assert mock_thread_class.call_count == 3
            for mock_thread in mock_thread_instances:
                mock_thread.start.assert_called_once()

    def test_pipeline_stop_stops_components(self, project_id, mock_source, mock_processor, mock_sink):
        """Test Pipeline stops all components in correct order."""
        pipeline = Pipeline(
            project_id=project_id,
            source=mock_source,
            processor=mock_processor,
            sink=mock_sink,
        )

        with patch("core.runtime.pipeline.Thread"):
            pipeline.start()

        pipeline.stop()

        mock_source.stop.assert_called_once()
        mock_processor.stop.assert_called_once()
        mock_sink.stop.assert_called_once()

    @pytest.mark.parametrize(
        "component_class",
        [Source, Processor, Sink],
    )
    def test_update_component_replaces_correct_type(
        self, project_id, mock_source, mock_processor, mock_sink, component_class
    ):
        """Test update_component replaces the correct component type."""
        pipeline = Pipeline(
            project_id=project_id,
            source=mock_source,
            processor=mock_processor,
            sink=mock_sink,
        )

        new_component = create_mock_component(component_class)

        with patch("core.runtime.pipeline.Thread"):
            pipeline.start()
            pipeline.update_component(new_component)

            # Check that only the matching component was stopped
            for component in (mock_source, mock_processor, mock_sink):
                if component.__class__ == component_class:
                    component.stop.assert_called_once()
                else:
                    component.stop.assert_not_called()

    def test_update_component_with_unknown_type_raises_error(self, project_id, mock_source, mock_processor, mock_sink):
        """Test update_component raises error for unknown component type."""
        pipeline = Pipeline(
            project_id=project_id,
            source=mock_source,
            processor=mock_processor,
            sink=mock_sink,
        )

        unknown_component = Mock()  # Not a known component type

        with pytest.raises(ValueError, match="Unknown component type"):
            pipeline.update_component(unknown_component)
