import queue
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from services.schemas.webrtc import Answer, Offer
from webrtc.manager import WebRTCManager


@pytest.fixture
def webrtc_manager():
    return WebRTCManager()


@pytest.fixture
def mock_pipeline():
    mock = MagicMock()
    mock.config.project_id = str(uuid4())
    mock.register_webrtc.return_value = queue.Queue()
    mock.unregister_webrtc = MagicMock()
    return mock


@pytest.mark.asyncio
async def test_handle_offer_success(webrtc_manager, mock_pipeline):
    """Test successful handling of WebRTC offer"""
    project_id = UUID(mock_pipeline.config.project_id)
    offer = Offer(webrtc_id="test-webrtc-id", sdp="v=0\r\no=- 123 456 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\n", type="offer")

    webrtc_manager.pm._pipeline = mock_pipeline

    with patch("webrtc.manager.RTCPeerConnection") as mock_rtc, patch("webrtc.manager.InferenceVideoStreamTrack"):
        mock_pc = MagicMock()  # Use MagicMock instead of AsyncMock
        mock_pc.localDescription = MagicMock(sdp="answer-sdp", type="answer")
        mock_pc.connectionState = "connected"
        mock_pc.setRemoteDescription = AsyncMock()
        mock_pc.createAnswer = AsyncMock()
        mock_pc.setLocalDescription = AsyncMock()
        mock_pc.addTrack = MagicMock()

        # Mock the decorator pattern for pc.on
        mock_pc.on = MagicMock(side_effect=lambda event: lambda func: func)

        mock_rtc.return_value = mock_pc

        answer = await webrtc_manager.handle_offer(project_id, offer)

        assert isinstance(answer, Answer)
        assert answer.sdp == "answer-sdp"
        assert answer.type == "answer"
        assert offer.webrtc_id in webrtc_manager._pcs
        assert webrtc_manager.queue is not None
        mock_pc.addTrack.assert_called_once()
        mock_pc.setRemoteDescription.assert_called_once()
        mock_pc.createAnswer.assert_called_once()
        mock_pc.setLocalDescription.assert_called_once()
        mock_pipeline.register_webrtc.assert_called_once()


@pytest.mark.asyncio
async def test_handle_offer_project_id_mismatch(webrtc_manager, mock_pipeline):
    """Test handling offer with mismatched project ID"""
    different_project_id = uuid4()
    offer = Offer(webrtc_id="test-webrtc-id", sdp="v=0\r\no=- 123 456 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\n", type="offer")

    webrtc_manager.pm._pipeline = mock_pipeline

    with pytest.raises(ValueError, match="Project ID does not match"):
        await webrtc_manager.handle_offer(different_project_id, offer)


@pytest.mark.asyncio
async def test_handle_offer_starts_pipeline_if_none(webrtc_manager):
    """Test that pipeline is started if not already running"""
    project_id = uuid4()
    offer = Offer(webrtc_id="test-webrtc-id", sdp="v=0\r\no=- 123 456 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\n", type="offer")

    mock_pipeline = MagicMock()
    mock_pipeline.config.project_id = str(project_id)
    mock_pipeline.register_webrtc.return_value = queue.Queue()

    webrtc_manager.pm.start = MagicMock()
    webrtc_manager.pm._pipeline = None

    with patch("webrtc.manager.RTCPeerConnection") as mock_rtc, patch("webrtc.manager.InferenceVideoStreamTrack"):
        mock_pc = MagicMock()  # Changed from AsyncMock to MagicMock
        mock_pc.localDescription = MagicMock(sdp="answer-sdp", type="answer")
        mock_pc.setRemoteDescription = AsyncMock()
        mock_pc.createAnswer = AsyncMock()
        mock_pc.setLocalDescription = AsyncMock()
        mock_pc.addTrack = MagicMock()

        # Mock the decorator pattern for pc.on
        mock_pc.on = MagicMock(side_effect=lambda event: lambda func: func)

        mock_rtc.return_value = mock_pc

        def start_side_effect():
            webrtc_manager.pm._pipeline = mock_pipeline

        webrtc_manager.pm.start.side_effect = start_side_effect

        answer = await webrtc_manager.handle_offer(project_id, offer)

        webrtc_manager.pm.start.assert_called_once()
        assert answer.sdp == "answer-sdp"


@pytest.mark.asyncio
async def test_handle_offer_connection_state_change_callback(webrtc_manager, mock_pipeline):
    """Test connection state change callback cleanup"""
    project_id = UUID(mock_pipeline.config.project_id)
    offer = Offer(webrtc_id="test-webrtc-id", sdp="v=0\r\no=- 123 456 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\n", type="offer")

    webrtc_manager.pm._pipeline = mock_pipeline

    with patch("webrtc.manager.RTCPeerConnection") as mock_rtc, patch("webrtc.manager.InferenceVideoStreamTrack"):
        mock_pc = AsyncMock()
        mock_pc.localDescription = MagicMock(sdp="answer-sdp", type="answer")
        mock_rtc.return_value = mock_pc

        connection_callback = None

        def capture_on_callback(event):
            def decorator(func):
                nonlocal connection_callback
                if event == "connectionstatechange":
                    connection_callback = func
                return func

            return decorator

        mock_pc.on = capture_on_callback

        await webrtc_manager.handle_offer(project_id, offer)

        # Simulate connection failure
        mock_pc.connectionState = "failed"
        await connection_callback()

        assert offer.webrtc_id not in webrtc_manager._pcs
        mock_pipeline.unregister_webrtc.assert_called_once()
