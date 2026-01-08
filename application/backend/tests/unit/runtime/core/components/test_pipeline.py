import time
from queue import Queue
from threading import Thread
from unittest.mock import Mock, PropertyMock, patch
from uuid import UUID, uuid4

import pytest

from domain.repositories.frame import FrameRepository
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


@pytest.fixture
def mock_frame_repository():
    return Mock(spec=FrameRepository)


class TestPipeline:
    def test_pipeline_initialization_with_no_components(self, project_id, mock_frame_repository):
        pipeline = Pipeline(project_id=project_id, frame_repository=mock_frame_repository)
        assert pipeline.project_id == project_id
        assert pipeline._components == {}
        assert pipeline.is_running is False
        pipeline.stop()

    def test_is_running_is_false_after_initialization(self, project_id, mock_frame_repository):
        pipeline = Pipeline(project_id=project_id, frame_repository=mock_frame_repository)
        assert pipeline.is_running is False
        pipeline.stop()

    def test_is_running_is_true_after_start(
        self,
        project_id,
        mock_source,
        mock_processor,
        mock_sink,
        mock_inbound_broadcaster,
        mock_outbound_broadcaster,
        mock_frame_repository,
    ):
        pipeline = (
            Pipeline(
                project_id=project_id,
                frame_repository=mock_frame_repository,
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

            assert pipeline.is_running is True

        pipeline.stop()

    def test_is_running_is_false_after_stop(
        self,
        project_id,
        mock_source,
        mock_processor,
        mock_sink,
        mock_inbound_broadcaster,
        mock_outbound_broadcaster,
        mock_frame_repository,
    ):
        pipeline = (
            Pipeline(
                project_id=project_id,
                frame_repository=mock_frame_repository,
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
            assert pipeline.is_running is True

            pipeline.stop()
            assert pipeline.is_running is False

    def test_set_source_registers_component(
        self, project_id, mock_source, mock_inbound_broadcaster, mock_frame_repository
    ):
        """Test that set_source registers the source component."""
        pipeline = Pipeline(
            project_id=project_id,
            frame_repository=mock_frame_repository,
            inbound_broadcaster=mock_inbound_broadcaster,
        )

        result = pipeline.set_source(mock_source)

        assert result is pipeline
        assert Source in pipeline._components
        assert pipeline._components[Source] == mock_source
        mock_source.setup.assert_called_once_with(mock_inbound_broadcaster)
        pipeline.stop()

    def test_set_processor_registers_component(
        self,
        project_id,
        mock_processor,
        mock_inbound_broadcaster,
        mock_outbound_broadcaster,
        mock_frame_repository,
    ):
        """Test that set_processor registers the processor component."""
        pipeline = Pipeline(
            project_id=project_id,
            frame_repository=mock_frame_repository,
            inbound_broadcaster=mock_inbound_broadcaster,
            outbound_broadcaster=mock_outbound_broadcaster,
        )

        result = pipeline.set_processor(mock_processor)

        assert result is pipeline
        assert Processor in pipeline._components
        assert pipeline._components[Processor] == mock_processor
        mock_processor.setup.assert_called_once_with(mock_inbound_broadcaster, mock_outbound_broadcaster)
        pipeline.stop()

    def test_set_sink_registers_component(
        self, project_id, mock_sink, mock_outbound_broadcaster, mock_frame_repository
    ):
        """Test that set_sink registers the sink component."""
        pipeline = Pipeline(
            project_id=project_id,
            frame_repository=mock_frame_repository,
            outbound_broadcaster=mock_outbound_broadcaster,
        )

        result = pipeline.set_sink(mock_sink)

        assert result is pipeline
        assert Sink in pipeline._components
        assert pipeline._components[Sink] == mock_sink
        mock_sink.setup.assert_called_once_with(mock_outbound_broadcaster)
        pipeline.stop()

    def test_pipeline_registers_and_unregisters_webrtc_consumer(
        self, project_id, mock_outbound_broadcaster, mock_frame_repository
    ):
        """Test registering and unregistering WebRTC consumers."""
        pipeline = Pipeline(
            project_id=project_id,
            frame_repository=mock_frame_repository,
            outbound_broadcaster=mock_outbound_broadcaster,
        )

        webrtc_queue = pipeline.register_webrtc()
        mock_outbound_broadcaster.register.assert_called_once()
        assert isinstance(webrtc_queue, Queue)

        pipeline.unregister_webrtc(webrtc_queue)
        mock_outbound_broadcaster.unregister.assert_called_once_with(queue=webrtc_queue)
        pipeline.stop()

    def test_pipeline_start_creates_threads_for_all_components(
        self,
        project_id,
        mock_source,
        mock_processor,
        mock_sink,
        mock_inbound_broadcaster,
        mock_outbound_broadcaster,
        mock_frame_repository,
    ):
        """Test that start() creates threads for all registered components."""
        pipeline = (
            Pipeline(
                project_id=project_id,
                frame_repository=mock_frame_repository,
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
        self,
        project_id,
        mock_source,
        mock_processor,
        mock_sink,
        mock_inbound_broadcaster,
        mock_outbound_broadcaster,
        mock_frame_repository,
    ):
        """Test that stop() stops all components."""
        pipeline = (
            Pipeline(
                project_id=project_id,
                frame_repository=mock_frame_repository,
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

    def test_start_does_nothing_if_already_running(
        self,
        project_id,
        mock_source,
        mock_processor,
        mock_sink,
        mock_inbound_broadcaster,
        mock_outbound_broadcaster,
        mock_frame_repository,
    ):
        pipeline = (
            Pipeline(
                project_id=project_id,
                frame_repository=mock_frame_repository,
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

            mock_thread_class.reset_mock()

            pipeline.start()
            assert mock_thread_class.call_count == 0
            assert pipeline.is_running is True

        pipeline.stop()

    def test_stop_does_nothing_if_already_stopped(
        self,
        project_id,
        mock_source,
        mock_processor,
        mock_sink,
        mock_inbound_broadcaster,
        mock_outbound_broadcaster,
        mock_frame_repository,
    ):
        pipeline = (
            Pipeline(
                project_id=project_id,
                frame_repository=mock_frame_repository,
                inbound_broadcaster=mock_inbound_broadcaster,
                outbound_broadcaster=mock_outbound_broadcaster,
            )
            .set_source(mock_source)
            .set_processor(mock_processor)
            .set_sink(mock_sink)
        )

        assert pipeline.is_running is False
        pipeline.stop()

        mock_source.stop.assert_not_called()
        mock_processor.stop.assert_not_called()
        mock_sink.stop.assert_not_called()

    def test_get_frame_index_waits_for_switch(self, project_id, mock_frame_repository):
        pipeline = Pipeline(project_id=project_id, frame_repository=mock_frame_repository)

        old_source = Mock(spec=Source)
        old_source.index.return_value = 5
        old_source.stop = Mock()
        old_source.setup = Mock()

        new_source = Mock(spec=Source)
        new_source.index.return_value = 10
        new_source.stop = Mock()
        new_source.setup = Mock()

        def slow_stop():
            time.sleep(0.5)

        old_source.stop.side_effect = slow_stop

        pipeline.set_source(old_source)

        def switch_source():
            pipeline.set_source(new_source)

        switch_thread = Thread(target=switch_source)

        switch_thread.start()
        time.sleep(0.1)
        idx = pipeline.get_frame_index()
        switch_thread.join()

        assert idx == 10

    def test_capture_frame_success(self, project_id, mock_inbound_broadcaster, mock_frame_repository):
        pipeline = Pipeline(
            project_id=project_id,
            frame_repository=mock_frame_repository,
            inbound_broadcaster=mock_inbound_broadcaster,
        )

        mock_input_data = Mock()
        mock_input_data.frame = Mock()
        type(mock_inbound_broadcaster).latest_frame = PropertyMock(return_value=mock_input_data)

        frame_id = pipeline.capture_frame()

        assert isinstance(frame_id, UUID)
        # Check that save_frame was called with the project_id, the frame_id, and the frame data
        mock_frame_repository.save_frame.assert_called_once_with(project_id, frame_id, mock_input_data.frame)

    def test_capture_frame_raises_error_if_no_frame(self, project_id, mock_inbound_broadcaster, mock_frame_repository):
        pipeline = Pipeline(
            project_id=project_id,
            frame_repository=mock_frame_repository,
            inbound_broadcaster=mock_inbound_broadcaster,
        )

        type(mock_inbound_broadcaster).latest_frame = PropertyMock(return_value=None)

        with pytest.raises(RuntimeError, match="No frame available from source"):
            pipeline.capture_frame()
