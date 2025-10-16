# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import queue
from uuid import UUID

from aiortc import RTCPeerConnection, RTCSessionDescription

from core.runtime.pipeline_manager import PipelineManager
from services.schemas.webrtc import Answer, Offer
from webrtc.stream import InferenceVideoStreamTrack

logger = logging.getLogger(__name__)


class WebRTCManager:
    """Manager for handling WebRTC connections."""

    def __init__(self, pipeline_manager: PipelineManager) -> None:
        self._pcs: dict[str, dict[str, RTCPeerConnection | queue.Queue]] = {}
        self.pipeline_manager = pipeline_manager

    async def handle_offer(self, project_id: UUID, offer: Offer) -> Answer:
        """Create an SDP offer for a new WebRTC connection."""
        pc = RTCPeerConnection()

        # compare projects_id from request with active pipeline project_id
        if str(project_id) != str(self.pipeline_manager.get_active_project_id()):
            raise ValueError("Project ID does not match the active pipeline's project ID.")

        # use PipelineManager to get queue
        rtc_queue = self.pipeline_manager.register_webrtc()

        # Store both connection and queue together
        self._pcs[offer.webrtc_id] = {"connection": pc, "queue": rtc_queue}

        # Add video track
        track = InferenceVideoStreamTrack(rtc_queue)
        pc.addTrack(track)

        @pc.on("connectionstatechange")
        async def connection_state_change() -> None:
            if pc.connectionState in ["failed", "closed"]:
                await self.cleanup_connection(offer.webrtc_id)
                self.pipeline_manager.unregister_webrtc(rtc_queue)

        # Set remote description from client's offer
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))

        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return Answer(sdp=pc.localDescription.sdp, type=pc.localDescription.type)

    @staticmethod
    async def _cleanup_pc_data(pc_data: dict[str, RTCPeerConnection | queue.Queue]) -> None:
        """Helper method to clean up a single connection's data."""
        queue_obj = pc_data.get("queue")
        if isinstance(queue_obj, queue.Queue):
            queue_obj.shutdown()

        connection_obj = pc_data.get("connection")
        if isinstance(connection_obj, RTCPeerConnection):
            await connection_obj.close()

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
