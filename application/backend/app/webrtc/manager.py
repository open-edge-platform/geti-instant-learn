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
        self._pcs[offer.webrtc_id] = {"connection": pc}

        # compare projects_id from request with active pipeline project_id
        if str(project_id) != str(self.pipeline_manager.get_active_project_id()):
            raise ValueError("Project ID does not match the active pipeline's project ID.")
        # use PipelineManager to get active pipeline and get queue
        self._pcs[offer.webrtc_id] = {"queue": self.pipeline_manager.register_webrtc()}
        rtc_queue = self._pcs[offer.webrtc_id]["queue"]

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

    async def cleanup_connection(self, webrtc_id: str) -> None:
        """Clean up a specific WebRTC connection by its ID."""
        if webrtc_id in self._pcs:
            logger.debug("Cleaning up connection: %s", webrtc_id)
            pc_data = self._pcs.pop(webrtc_id)
            queue_obj = pc_data["queue"]
            connection_obj = pc_data["connection"]

            if isinstance(queue_obj, queue.Queue):
                queue_obj.shutdown()
            if isinstance(connection_obj, RTCPeerConnection):
                await connection_obj.close()
            logger.debug("Connection %s successfully closed.", webrtc_id)

    async def cleanup(self) -> None:
        """Clean up all connections"""
        for pc_data in list(self._pcs.values()):
            queue_obj = pc_data["queue"]
            connection_obj = pc_data["connection"]

            if isinstance(queue_obj, queue.Queue):
                queue_obj.shutdown()
            if isinstance(connection_obj, RTCPeerConnection):
                await connection_obj.close()
        self._pcs.clear()
