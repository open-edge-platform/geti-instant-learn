# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from collections.abc import Callable

import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame

from domain.services.schemas.label import VisualizationInfo
from domain.services.schemas.processor import OutputData
from runtime.core.components.broadcaster import FrameSlot
from runtime.webrtc.visualizer import InferenceVisualizer

logger = logging.getLogger(__name__)

FALLBACK_FRAME = np.full((64, 64, 3), 16, dtype=np.uint8)


class InferenceVideoStreamTrack(VideoStreamTrack):
    """A video stream track that provides frames with inference results over WebRTC.

    Reads the latest processed frame from a shared FrameSlot rather than
    consuming from a queue.  Because ``recv()`` is called at ~30 fps by
    aiortc, the same frame may be returned multiple times until the pipeline
    publishes a new one.  Visualization and tracing are applied only once per
    unique frame.
    """

    def __init__(
        self,
        output_slot: FrameSlot[OutputData],
        enable_visualization: bool = True,
        visualization_info_provider: Callable[[], VisualizationInfo | None] | None = None,
    ):
        super().__init__()
        self._slot = output_slot
        self._last_output: OutputData | None = None
        self._last_frame: np.ndarray | None = None
        self._enable_visualization = enable_visualization
        self._visualizer = InferenceVisualizer(enable_visualization)
        self._visualization_info_provider = visualization_info_provider

    async def recv(self) -> VideoFrame:
        """Return the next video frame for WebRTC streaming.

        Reads ``self._slot.latest`` on every call.  When a new
        ``OutputData`` is detected (identity check), visualization and
        tracing are applied and the rendered numpy array is cached.
        Subsequent calls that see the same ``OutputData`` reuse the cache.

        Falls back to a small dark-gray placeholder when no frame has been
        published yet.
        """
        pts, time_base = await self.next_timestamp()
        wall_ms = time.monotonic() * 1000.0
        pts_ms = pts * float(time_base) * 1000.0 if time_base else 0.0

        output_data = self._slot.latest

        if output_data is not None and output_data is not self._last_output:
            # New frame from the pipeline — visualize and cache
            self._last_output = output_data

            if output_data.trace:
                output_data.trace.record_start("webrtc")

            if self._enable_visualization and self._visualizer:
                vis_info = self._visualization_info_provider() if self._visualization_info_provider else None
                np_frame = self._visualizer.visualize(output_data=output_data, visualization_info=vis_info)
            else:
                np_frame = output_data.frame

            self._last_frame = np_frame

            if output_data.trace:
                output_data.trace.record_end("webrtc")
                logger.info(output_data.trace.format_log())

        np_frame = self._last_frame if self._last_frame is not None else FALLBACK_FRAME

        logger.debug("pts=%.2f ms | wall=%.2f ms | drift=%.2f ms", pts_ms, wall_ms, wall_ms - pts_ms)
        frame = VideoFrame.from_ndarray(np_frame, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame
