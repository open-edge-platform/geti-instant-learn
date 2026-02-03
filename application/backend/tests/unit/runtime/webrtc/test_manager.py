from queue import Queue
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from aiortc import RTCPeerConnection, RTCSessionDescription

from domain.services.schemas.label import CategoryMappings, RGBColor, VisualizationInfo, VisualizationLabel
from domain.services.schemas.webrtc import Answer, Offer
from runtime.errors import PipelineProjectMismatchError
from runtime.pipeline_manager import PipelineManager
from runtime.webrtc.manager import WebRTCManager
from runtime.webrtc.sdp_handler import SDPHandler

PROJECT_ID = uuid4()


@pytest.fixture
def mock_pipeline_manager():
    """Create a mock PipelineManager."""
    pm = Mock(spec=PipelineManager)
    pm.register_webrtc.return_value = Queue()
    pm.unregister_webrtc.return_value = None
    return pm


@pytest.fixture
def mock_sdp_handler():
    """Create a mock SDPHandler."""
    handler = Mock(spec=SDPHandler)
    handler.mangle_sdp = AsyncMock(return_value="mangled-sdp")
    return handler


@pytest.fixture
def webrtc_manager(mock_pipeline_manager, mock_sdp_handler):
    """Create a WebRTCManager instance with mocked dependencies."""
    return WebRTCManager(pipeline_manager=mock_pipeline_manager, sdp_handler=mock_sdp_handler)


@pytest.fixture
def sample_offer():
    """Create a sample Offer object."""
    return Offer(webrtc_id="test-webrtc-id", sdp="v=0\r\n", type="offer")


def _make_vis_info() -> VisualizationInfo:
    return VisualizationInfo(
        label_colors=[VisualizationLabel(id=uuid4(), color=RGBColor(1, 2, 3), object_name=None)],
        category_mappings=CategoryMappings(label_to_category_id={}, category_id_to_label_id={}),
    )


@pytest.mark.asyncio
async def test_handle_offer_success(webrtc_manager, mock_pipeline_manager, sample_offer):
    """Test successful offer handling with matching project IDs."""
    with (
        patch("runtime.webrtc.manager.RTCPeerConnection") as MockRTCPeerConnection,
        patch("runtime.webrtc.manager.InferenceVideoStreamTrack") as MockTrack,
    ):
        mock_pc = AsyncMock(spec=RTCPeerConnection)
        mock_pc.localDescription = Mock(sdp="answer-sdp", type="answer")
        mock_pc.connectionState = "active"
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
        patch("runtime.webrtc.manager.RTCPeerConnection") as MockRTCPeerConnection,
        patch("runtime.webrtc.manager.InferenceVideoStreamTrack") as MockTrack,
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
        patch("runtime.webrtc.manager.RTCPeerConnection") as MockRTCPeerConnection,
        patch("runtime.webrtc.manager.InferenceVideoStreamTrack") as MockTrack,
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
        patch("runtime.webrtc.manager.RTCPeerConnection") as MockRTCPeerConnection,
        patch("runtime.webrtc.manager.InferenceVideoStreamTrack") as MockTrack,
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
async def test_connection_state_change_triggers_cleanup(webrtc_manager, mock_pipeline_manager, sample_offer):
    """Test that connection state change to 'failed' or 'closed' triggers cleanup."""
    with (
        patch("runtime.webrtc.manager.RTCPeerConnection") as MockRTCPeerConnection,
        patch("runtime.webrtc.manager.InferenceVideoStreamTrack") as MockTrack,
        patch("runtime.webrtc.manager.isinstance", return_value=True),
    ):
        # Setup mocks
        mock_track = Mock()
        mock_track.kind = "video"
        MockTrack.return_value = mock_track

        mock_pc = AsyncMock(spec=RTCPeerConnection)
        mock_pc.localDescription = Mock(sdp="answer-sdp", type="answer")
        mock_pc.connectionState = "active"
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

        mock_pc.connectionState = "closed"

        # Trigger the callback
        assert captured_callback is not None
        await captured_callback()

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
        patch("runtime.webrtc.manager.RTCPeerConnection") as MockRTCPeerConnection,
        patch("runtime.webrtc.manager.InferenceVideoStreamTrack") as MockTrack,
        patch("runtime.webrtc.manager.isinstance", return_value=True),
    ):
        # Setup mocks
        mock_track = Mock()
        mock_track.kind = "video"
        MockTrack.return_value = mock_track

        mock_pcs = []
        for _ in range(num_connections):
            mock_pc = AsyncMock(spec=RTCPeerConnection)
            mock_pc.localDescription = Mock(sdp="answer-sdp", type="answer")
            mock_pc.connectionState = "active"
            mock_pcs.append(mock_pc)

        MockRTCPeerConnection.side_effect = mock_pcs

        # Create all connections
        for offer in offers:
            await webrtc_manager.handle_offer(PROJECT_ID, offer)

        # Get all queues and mock their shutdown methods
        queues = []
        for conn_data in webrtc_manager._pcs.values():
            queue = conn_data.queue
            queues.append(queue)

        # Execute cleanup
        await webrtc_manager.cleanup()

        # Verify all connections removed
        assert len(webrtc_manager._pcs) == 0

        # Verify close was called on each peer connection
        for mock_pc in mock_pcs:
            mock_pc.close.assert_called_once()


@pytest.mark.asyncio
async def test_handle_offer_with_hostname_resolution(webrtc_manager, mock_pipeline_manager, sample_offer):
    with (
        patch("runtime.webrtc.manager.RTCPeerConnection") as MockRTCPeerConnection,
        patch("runtime.webrtc.manager.InferenceVideoStreamTrack") as MockTrack,
        patch("runtime.webrtc.manager.get_settings") as mock_get_settings,
    ):
        # Setup settings
        mock_settings = Mock()
        mock_settings.ice_servers = []
        mock_settings.webrtc_advertise_ip = "my-domain.com"
        mock_get_settings.return_value = mock_settings

        # Setup mocks
        mock_pc = AsyncMock(spec=RTCPeerConnection)
        mock_pc.localDescription = Mock(sdp="original-sdp", type="answer")
        MockRTCPeerConnection.return_value = mock_pc
        MockTrack.return_value = Mock()

        # Configure SDP handler mock
        webrtc_manager.sdp_handler.mangle_sdp.return_value = "mangled-sdp"

        answer = await webrtc_manager.handle_offer(PROJECT_ID, sample_offer)

        # Verify mangling was called with domain name (resolution happens inside handler)
        webrtc_manager.sdp_handler.mangle_sdp.assert_called_once_with("original-sdp", "my-domain.com")

        assert answer.sdp == "mangled-sdp"


def test_get_visualization_info_returns_value(webrtc_manager, mock_pipeline_manager) -> None:
    vis_info = _make_vis_info()
    mock_pipeline_manager.get_visualization_info.return_value = vis_info

    result = webrtc_manager.get_visualization_info(PROJECT_ID)

    assert result is vis_info
    mock_pipeline_manager.get_visualization_info.assert_called_once_with(PROJECT_ID)


def test_get_visualization_info_returns_none_on_error(webrtc_manager, mock_pipeline_manager) -> None:
    mock_pipeline_manager.get_visualization_info.side_effect = RuntimeError("db error")

    result = webrtc_manager.get_visualization_info(PROJECT_ID)

    assert result is None
    mock_pipeline_manager.get_visualization_info.assert_called_once_with(PROJECT_ID)
