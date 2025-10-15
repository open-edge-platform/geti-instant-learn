# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from aiortc import RTCPeerConnection, RTCSessionDescription

from core.runtime.dispatcher import ConfigChangeDispatcher
from core.runtime.pipeline_manager import DummyProjectRepo, PipelineManager
from services.schemas.webrtc import Answer, InputData, Offer
from webrtc.stream import InferenceVideoStreamTrack

if TYPE_CHECKING:
    import queue

logger = logging.getLogger(__name__)


class WebRTCManager:
    """Manager for handling WebRTC connections."""

    def __init__(self) -> None:
        self._pcs: dict[str, RTCPeerConnection] = {}
        self._input_data: dict[str, Any] = {}
        self.queue: queue.Queue | None = None
        self.pm = PipelineManager(event_dispatcher=ConfigChangeDispatcher(), project_repo=DummyProjectRepo())

    async def handle_offer(self, project_id: UUID, offer: Offer) -> Answer:
        """Create an SDP offer for a new WebRTC connection."""
        pc = RTCPeerConnection()
        self._pcs[offer.webrtc_id] = pc

        # use PipelineManager to get active pipeline and get queue
        pipeline = self.pm.get_or_start_pipeline()
        self.queue = pipeline.register_webrtc()
        # compare projects_id from request with active pipeline project_id
        if str(project_id) != str(pipeline.config.project_id):
            raise ValueError("Project ID does not match the active pipeline's project ID.")

        # Add video track
        track = InferenceVideoStreamTrack(self.queue)
        pc.addTrack(track)

        @pc.on("connectionstatechange")
        async def connection_state_change() -> None:
            if pc.connectionState in ["failed", "closed"]:
                await self.cleanup_connection(offer.webrtc_id)
                pipeline.unregister_webrtc(self.queue)

        # Set remote description from client's offer
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))

        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return Answer(sdp=pc.localDescription.sdp, type=pc.localDescription.type)

    def set_input(self, project_id: UUID, data: InputData) -> None:
        """Set input data for specific WebRTC connection"""
        if str(project_id) == str(data.project_id):
            self._input_data[data.webrtc_id] = {
                "conf_threshold": data.conf_threshold,
                "updated_at": asyncio.get_event_loop().time(),
            }

    async def cleanup_connection(self, webrtc_id: str) -> None:
        """Clean up a specific WebRTC connection by its ID."""
        if webrtc_id in self._pcs:
            logger.debug("Cleaning up connection: %s", webrtc_id)
            pc = self._pcs.pop(webrtc_id)
            await pc.close()
            logger.debug("Connection %s successfully closed.", webrtc_id)
            self._input_data.pop(webrtc_id, None)

    async def cleanup(self) -> None:
        """Clean up all connections"""
        for pc in list(self._pcs.values()):
            await pc.close()
        self._pcs.clear()
        self._input_data.clear()
