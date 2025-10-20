# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from queue import Queue
from uuid import UUID

from aiortc import RTCPeerConnection, RTCSessionDescription

from core.runtime.errors import PipelineNotActiveError, PipelineProjectMismatchError
from core.runtime.pipeline_manager import PipelineManager
from services.schemas.webrtc import Answer, Offer
from webrtc.stream import InferenceVideoStreamTrack

logger = logging.getLogger(__name__)


@dataclass
class ConnectionData:
    connection: RTCPeerConnection
    queue: Queue


class WebRTCManager:
    """Manager for handling WebRTC connections."""

    def __init__(self, pipeline_manager: PipelineManager) -> None:
        self._pcs: dict[str, ConnectionData] = {}
        self.pipeline_manager = pipeline_manager

    async def handle_offer(self, project_id: UUID, offer: Offer) -> Answer:
        """Create an SDP offer for a new WebRTC connection."""
        pc = RTCPeerConnection()

        # use PipelineManager to get queue
        try:
            rtc_queue = self.pipeline_manager.register_webrtc(project_id=project_id)
        except PipelineProjectMismatchError:
            logger.exception(f"Failed to register WebRTC for project {project_id}")
            raise
        except PipelineNotActiveError as e:
            logger.exception(f"Pipeline not active for project {project_id}: {e}")
            raise

        # Store both connection and queue together
        self._pcs[offer.webrtc_id] = ConnectionData(connection=pc, queue=rtc_queue)

        # Add video track
        track = InferenceVideoStreamTrack(rtc_queue)
        pc.addTrack(track)

        @pc.on("connectionstatechange")
        async def connection_state_change() -> None:
            if pc.connectionState in ["failed", "closed"]:
                await self.cleanup_connection(offer.webrtc_id)
                try:
                    self.pipeline_manager.unregister_webrtc(rtc_queue, project_id=project_id)
                except PipelineProjectMismatchError as e:
                    logger.exception(f"Failed to unregister WebRTC for project {project_id}: {e}")
                    raise
                except PipelineNotActiveError as e:
                    logger.exception(f"Pipeline not active for project {project_id}: {e}")
                    raise

        # Set remote description from client's offer
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))

        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return Answer(sdp=pc.localDescription.sdp, type=pc.localDescription.type)

    @staticmethod
    async def _cleanup_pc_data(pc_data: ConnectionData) -> None:
        """Helper method to clean up a single connection's data."""
        if isinstance(pc_data.queue, Queue):
            pc_data.queue.shutdown()

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
