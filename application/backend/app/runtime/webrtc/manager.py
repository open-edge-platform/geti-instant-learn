# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from queue import Queue
from uuid import UUID

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription

from domain.services.schemas.webrtc import Answer, Offer
from runtime.errors import PipelineNotActiveError, PipelineProjectMismatchError
from runtime.pipeline_manager import PipelineManager
from runtime.webrtc.sdp_handler import SDPHandler
from runtime.webrtc.stream import InferenceVideoStreamTrack
from settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ConnectionData:
    connection: RTCPeerConnection
    queue: Queue


class WebRTCManager:
    """Manager for handling WebRTC connections."""

    def __init__(self, pipeline_manager: PipelineManager, sdp_handler: SDPHandler) -> None:
        self._pcs: dict[str, ConnectionData] = {}
        self.pipeline_manager = pipeline_manager
        self.sdp_handler = sdp_handler

    async def handle_offer(self, project_id: UUID, offer: Offer) -> Answer:
        """Create an SDP offer for a new WebRTC connection."""
        settings = get_settings()
        ice_servers = [RTCIceServer(**server) for server in settings.ice_servers]
        config = RTCConfiguration(iceServers=ice_servers)
        pc = RTCPeerConnection(configuration=config)

        # use PipelineManager to get queue
        try:
            rtc_queue = self.pipeline_manager.register_webrtc(project_id=project_id)
        except (PipelineProjectMismatchError, PipelineNotActiveError) as exc:
            logger.exception(f"Failed to register WebRTC for project {project_id}: {exc}")
            raise

        # Store both connection and queue together
        self._pcs[offer.webrtc_id] = ConnectionData(connection=pc, queue=rtc_queue)

        # Add video track
        track = InferenceVideoStreamTrack(stream_queue=rtc_queue, enable_visualization=True)
        pc.addTrack(track)

        @pc.on("connectionstatechange")
        async def connection_state_change() -> None:
            if pc.connectionState in ["failed", "closed"]:
                try:
                    # First unregister from pipeline manager (stops broadcasting to this queue)
                    self.pipeline_manager.unregister_webrtc(rtc_queue, project_id=project_id)
                except (PipelineProjectMismatchError, PipelineNotActiveError) as exc:
                    logger.exception(f"Failed to unregister WebRTC for project {project_id}: {exc}")
                    raise
                finally:
                    # Then cleanup the connection (shuts down the queue)
                    await self.cleanup_connection(offer.webrtc_id)

        # Set remote description from client's offer
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))

        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # Mangle SDP if public IP is configured
        sdp = pc.localDescription.sdp
        if settings.webrtc_advertise_ip:
            sdp = await self.sdp_handler.mangle_sdp(sdp, settings.webrtc_advertise_ip)

        return Answer(sdp=sdp, type=pc.localDescription.type)

    @staticmethod
    async def _cleanup_pc_data(pc_data: ConnectionData) -> None:
        """Helper method to clean up a single connection's data."""
        if isinstance(pc_data.connection, RTCPeerConnection):
            await pc_data.connection.close()

    async def cleanup_connection(self, webrtc_id: str) -> None:
        """Clean up a specific WebRTC connection by its ID."""
        pc_data = self._pcs.pop(webrtc_id, None)
        if pc_data:
            logger.debug("Cleaning up connection: %s", webrtc_id)
            await self._cleanup_pc_data(pc_data)
            logger.debug("Connection %s successfully closed.", webrtc_id)

    async def cleanup(self) -> None:
        """Clean up all connections"""
        for pc_data in list(self._pcs.values()):
            await self._cleanup_pc_data(pc_data)
        self._pcs.clear()
