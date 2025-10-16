import queue
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from aiortc import RTCPeerConnection, RTCSessionDescription

from core.runtime.pipeline_manager import PipelineManager
from services.schemas.webrtc import Answer, Offer
from webrtc.manager import WebRTCManager

PROJECT_ID = uuid4()


@pytest.fixture
def mock_pipeline_manager():
    """Create a mock PipelineManager."""
    pm = Mock(spec=PipelineManager)
    pm.get_project_id.return_value = PROJECT_ID
    pm.register_webrtc.return_value = queue.Queue()
    pm.unregister_webrtc.return_value = None
    return pm


@pytest.fixture
def webrtc_manager(mock_pipeline_manager):
    """Create a WebRTCManager instance with mocked dependencies."""
    return WebRTCManager(pipeline_manager=mock_pipeline_manager)


@pytest.fixture
def sample_offer():
    """Create a sample Offer object."""
    return Offer(webrtc_id="test-webrtc-id", sdp="v=0\r\no=- 123456789 2 IN IP4 127.0.0.1\r\n", type="offer")


@pytest.mark.asyncio
async def test_handle_offer_success(webrtc_manager, mock_pipeline_manager, sample_offer):
    """Test successful offer handling with matching project IDs."""
    project_id = PROJECT_ID

    with patch("webrtc.manager.RTCPeerConnection") as MockRTCPeerConnection:
        mock_pc = AsyncMock(spec=RTCPeerConnection)
        mock_pc.localDescription = Mock(sdp="answer-sdp", type="answer")
        mock_pc.connectionState = "connected"
        MockRTCPeerConnection.return_value = mock_pc

        answer = await webrtc_manager.handle_offer(project_id, sample_offer)

        assert isinstance(answer, Answer)
        assert answer.sdp == "answer-sdp"
        assert answer.type == "answer"
        assert sample_offer.webrtc_id in webrtc_manager._pcs
        mock_pipeline_manager.register_webrtc.assert_called_once()


@pytest.mark.asyncio
async def test_handle_offer_project_id_mismatch(webrtc_manager, mock_pipeline_manager, sample_offer):
    """Test offer handling fails when project IDs don't match."""
    wrong_project_id = uuid4()

    with pytest.raises(ValueError, match="Project ID does not match"):
        await webrtc_manager.handle_offer(wrong_project_id, sample_offer)

    mock_pipeline_manager.register_webrtc.assert_not_called()


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

    with patch("webrtc.manager.RTCPeerConnection") as MockRTCPeerConnection:
        mock_pc = AsyncMock(spec=RTCPeerConnection)
        mock_pc.localDescription = Mock(sdp="answer-sdp", type="answer")
        MockRTCPeerConnection.return_value = mock_pc

        await webrtc_manager.handle_offer(PROJECT_ID, sample_offer)

        mock_pc.on.assert_called_with("connectionstatechange")


@pytest.mark.asyncio
async def test_handle_offer_sets_remote_description(webrtc_manager, mock_pipeline_manager, sample_offer):
    """Test that remote description is set from the offer."""

    with patch("webrtc.manager.RTCPeerConnection") as MockRTCPeerConnection:
        mock_pc = AsyncMock(spec=RTCPeerConnection)
        mock_pc.localDescription = Mock(sdp="answer-sdp", type="answer")
        MockRTCPeerConnection.return_value = mock_pc

        await webrtc_manager.handle_offer(PROJECT_ID, sample_offer)

        mock_pc.setRemoteDescription.assert_called_once()
        call_args = mock_pc.setRemoteDescription.call_args[0][0]
        assert isinstance(call_args, RTCSessionDescription)
        assert call_args.sdp == sample_offer.sdp
        assert call_args.type == sample_offer.type
