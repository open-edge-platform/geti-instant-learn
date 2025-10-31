from queue import Queue
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from aiortc import RTCPeerConnection, RTCSessionDescription

from core.runtime.errors import PipelineProjectMismatchError
from core.runtime.pipeline_manager import PipelineManager
from services.schemas.webrtc import Answer, Offer
from webrtc.manager import WebRTCManager

PROJECT_ID = uuid4()


@pytest.fixture
def mock_pipeline_manager():
    """Create a mock PipelineManager."""
    pm = Mock(spec=PipelineManager)
    pm.register_webrtc.return_value = Queue()
    pm.unregister_webrtc.return_value = None
    return pm


@pytest.fixture
def webrtc_manager(mock_pipeline_manager):
    """Create a WebRTCManager instance with mocked dependencies."""
    return WebRTCManager(pipeline_manager=mock_pipeline_manager)


@pytest.fixture
def sample_offer():
    """Create a sample Offer object."""
    return Offer(webrtc_id="test-webrtc-id", sdp="v=0\r\n", type="offer")


@pytest.mark.asyncio
async def test_handle_offer_success(webrtc_manager, mock_pipeline_manager, sample_offer):
    """Test successful offer handling with matching project IDs."""
    with (
        patch("webrtc.manager.RTCPeerConnection") as MockRTCPeerConnection,
        patch("webrtc.manager.InferenceVideoStreamTrack") as MockTrack,
    ):
        mock_pc = AsyncMock(spec=RTCPeerConnection)
        mock_pc.localDescription = Mock(sdp="answer-sdp", type="answer")
        mock_pc.connectionState = "connected"
        MockRTCPeerConnection.return_value = mock_pc
        MockTrack.return_value = Mock()

        answer = await webrtc_manager.handle_offer(PROJECT_ID, sample_offer)

        assert isinstance(answer, Answer)
        assert answer.sdp == "answer-sdp"
        assert answer.type == "answer"
        assert sample_offer.webrtc_id in webrtc_manager._pcs
        mock_pipeline_manager.register_webrtc.assert_called_once()


@pytest.mark.asyncio
async def test_handle_offer_project_id_mismatch(webrtc_manager, mock_pipeline_manager, sample_offer):
    """Test offer handling fails when project IDs don't match."""
    wrong_project_id = uuid4()

    # Mock register_webrtc to raise exception on project mismatch
    mock_pipeline_manager.register_webrtc.side_effect = PipelineProjectMismatchError(
        f"Project ID mismatch: expected {PROJECT_ID}, got {wrong_project_id}"
    )

    with pytest.raises(PipelineProjectMismatchError):
        await webrtc_manager.handle_offer(wrong_project_id, sample_offer)

    mock_pipeline_manager.register_webrtc.assert_called_once()


@pytest.mark.asyncio
async def test_handle_offer_creates_video_track(webrtc_manager, mock_pipeline_manager, sample_offer):
    """Test that video track is added to the peer connection."""
    with (
        patch("webrtc.manager.RTCPeerConnection") as MockRTCPeerConnection,
        patch("webrtc.manager.InferenceVideoStreamTrack") as MockTrack,
    ):
        mock_pc = AsyncMock(spec=RTCPeerConnection)
        mock_pc.localDescription = Mock(sdp="answer-sdp", type="answer")
        MockRTCPeerConnection.return_value = mock_pc

        mock_track = Mock()
        MockTrack.return_value = mock_track

        await webrtc_manager.handle_offer(PROJECT_ID, sample_offer)

        mock_pc.addTrack.assert_called_once_with(mock_track)


@pytest.mark.asyncio
async def test_handle_offer_registers_connection_state_handler(webrtc_manager, mock_pipeline_manager, sample_offer):
    """Test that connection state change handler is registered."""
    with (
        patch("webrtc.manager.RTCPeerConnection") as MockRTCPeerConnection,
        patch("webrtc.manager.InferenceVideoStreamTrack") as MockTrack,
    ):
        mock_pc = AsyncMock(spec=RTCPeerConnection)
        mock_pc.localDescription = Mock(sdp="answer-sdp", type="answer")
        MockRTCPeerConnection.return_value = mock_pc
        MockTrack.return_value = Mock()

        await webrtc_manager.handle_offer(PROJECT_ID, sample_offer)

        mock_pc.on.assert_called_with("connectionstatechange")


@pytest.mark.asyncio
async def test_handle_offer_sets_remote_description(webrtc_manager, mock_pipeline_manager, sample_offer):
    """Test that remote description is set from the offer."""
    with (
        patch("webrtc.manager.RTCPeerConnection") as MockRTCPeerConnection,
        patch("webrtc.manager.InferenceVideoStreamTrack") as MockTrack,
    ):
        mock_pc = AsyncMock(spec=RTCPeerConnection)
        mock_pc.localDescription = Mock(sdp="answer-sdp", type="answer")
        MockRTCPeerConnection.return_value = mock_pc
        MockTrack.return_value = Mock()

        await webrtc_manager.handle_offer(PROJECT_ID, sample_offer)

        mock_pc.setRemoteDescription.assert_called_once()
        call_args = mock_pc.setRemoteDescription.call_args[0][0]
        assert isinstance(call_args, RTCSessionDescription)
        assert call_args.sdp == sample_offer.sdp
        assert call_args.type == sample_offer.type


@pytest.mark.asyncio
async def test_cleanup_connection_disposes_queue(webrtc_manager, mock_pipeline_manager, sample_offer):
    """Test that cleanup_connection properly disposes the queue."""
    with (
        patch("webrtc.manager.RTCPeerConnection") as MockRTCPeerConnection,
        patch("webrtc.manager.InferenceVideoStreamTrack") as MockTrack,
        patch("webrtc.manager.isinstance", return_value=True),
    ):
        # Setup mocks
        mock_track = Mock()
        mock_track.kind = "video"
        MockTrack.return_value = mock_track

        mock_pc = AsyncMock(spec=RTCPeerConnection)
        mock_pc.localDescription = Mock(sdp="answer-sdp", type="answer")
        mock_pc.connectionState = "connected"
        MockRTCPeerConnection.return_value = mock_pc

        # Create connection
        await webrtc_manager.handle_offer(PROJECT_ID, sample_offer)

        # Get connection data
        conn_data = webrtc_manager._pcs[sample_offer.webrtc_id]
        queue = conn_data.queue

        # Mock queue shutdown
        with patch.object(queue, "shutdown") as mock_shutdown:
            # Execute cleanup
            await webrtc_manager.cleanup_connection(sample_offer.webrtc_id)

            # Verify queue shutdown was called
            mock_shutdown.assert_called_once()

            # Verify connection close was called
            mock_pc.close.assert_called_once()

            # Verify connection removed from registry
            assert sample_offer.webrtc_id not in webrtc_manager._pcs

            # Note: unregister_webrtc is NOT called in cleanup_connection
            # It's only called in the connection_state_change callback
            # when the connection state becomes "failed" or "closed"


@pytest.mark.asyncio
async def test_connection_state_change_triggers_cleanup(webrtc_manager, mock_pipeline_manager, sample_offer):
    """Test that connection state change to 'failed' or 'closed' triggers cleanup."""
    with (
        patch("webrtc.manager.RTCPeerConnection") as MockRTCPeerConnection,
        patch("webrtc.manager.InferenceVideoStreamTrack") as MockTrack,
        patch("webrtc.manager.isinstance", return_value=True),
    ):
        # Setup mocks
        mock_track = Mock()
        mock_track.kind = "video"
        MockTrack.return_value = mock_track

        mock_pc = AsyncMock(spec=RTCPeerConnection)
        mock_pc.localDescription = Mock(sdp="answer-sdp", type="answer")
        mock_pc.connectionState = "connected"
        MockRTCPeerConnection.return_value = mock_pc

        # Capture the state change callback
        captured_callback = None

        def on_decorator(event):
            def wrapper(callback):
                nonlocal captured_callback
                if event == "connectionstatechange":
                    captured_callback = callback
                return callback

            return wrapper

        mock_pc.on.side_effect = on_decorator

        # Create connection
        await webrtc_manager.handle_offer(PROJECT_ID, sample_offer)

        # Get connection data
        conn_data = webrtc_manager._pcs[sample_offer.webrtc_id]
        queue = conn_data.queue

        # Mock queue shutdown
        with patch.object(queue, "shutdown") as mock_shutdown:
            # Simulate connection state change to closed
            mock_pc.connectionState = "closed"

            # Trigger the callback
            assert captured_callback is not None
            await captured_callback()

            # Verify queue shutdown was called
            mock_shutdown.assert_called_once()

            # Verify unregister was called
            mock_pipeline_manager.unregister_webrtc.assert_called_once_with(queue, project_id=PROJECT_ID)

            # Verify connection removed from registry
            assert sample_offer.webrtc_id not in webrtc_manager._pcs


@pytest.mark.asyncio
async def test_cleanup_all_connections(webrtc_manager, mock_pipeline_manager):
    """Test that cleanup() disposes all connections and queues."""
    # Create multiple offers
    num_connections = 3
    offers = [Offer(webrtc_id=f"conn-{i}", sdp="v=0\r\n", type="offer") for i in range(num_connections)]

    with (
        patch("webrtc.manager.RTCPeerConnection") as MockRTCPeerConnection,
        patch("webrtc.manager.InferenceVideoStreamTrack") as MockTrack,
        patch("webrtc.manager.isinstance", return_value=True),
    ):
        # Setup mocks
        mock_track = Mock()
        mock_track.kind = "video"
        MockTrack.return_value = mock_track

        mock_pcs = []
        for _ in range(num_connections):
            mock_pc = AsyncMock(spec=RTCPeerConnection)
            mock_pc.localDescription = Mock(sdp="answer-sdp", type="answer")
            mock_pc.connectionState = "connected"
            mock_pcs.append(mock_pc)

        MockRTCPeerConnection.side_effect = mock_pcs

        # Create all connections
        for offer in offers:
            await webrtc_manager.handle_offer(PROJECT_ID, offer)

        # Get all queues and mock their shutdown methods
        queues = []
        for conn_data in webrtc_manager._pcs.values():
            queue = conn_data.queue
            queue.shutdown = Mock()
            queues.append(queue)

        # Execute cleanup
        await webrtc_manager.cleanup()

        # Verify all connections removed
        assert len(webrtc_manager._pcs) == 0

        # Verify close was called on each peer connection
        for mock_pc in mock_pcs:
            mock_pc.close.assert_called_once()

        # Verify shutdown was called on each queue
        assert queue.shutdown.call_count == num_connections
