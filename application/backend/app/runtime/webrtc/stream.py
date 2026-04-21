# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import textwrap
from collections.abc import Callable

import cv2
import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame

from domain.services.schemas.label import VisualizationInfo
from domain.services.schemas.processor import OutputData
from runtime.core.components.broadcaster import FrameSlot
from runtime.webrtc.visualizer import InferenceVisualizer

logger = logging.getLogger(__name__)

FALLBACK_FRAME = np.full((64, 64, 3), 16, dtype=np.uint8)


def create_error_frame(error_message: str, width: int = 1280, height: int = 720) -> np.ndarray:
    """Create a frame with error text overlay.

    Args:
        error_message: The error message to display.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        RGB frame with error text overlay.
    """
    # Create dark background
    frame = np.full((height, width, 3), 32, dtype=np.uint8)

    # Wrap text to fit frame width
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    line_spacing = 40

    # Add title
    title = "Source Connection Error"
    title_size = cv2.getTextSize(title, font, font_scale * 1.2, thickness + 1)[0]
    title_x = (width - title_size[0]) // 2
    title_y = height // 3
    cv2.putText(frame, title, (title_x, title_y), font, font_scale * 1.2, (255, 100, 100), thickness + 1)

    # Wrap and display error message
    wrapped_lines = textwrap.wrap(error_message, width=60)
    y_offset = title_y + 60

    for line in wrapped_lines:
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        cv2.putText(frame, line, (text_x, y_offset), font, font_scale, (255, 255, 255), thickness)
        y_offset += line_spacing

    # Add instruction
    instruction = "Please check the source configuration and try again."
    inst_size = cv2.getTextSize(instruction, font, font_scale * 0.8, thickness - 1)[0]
    inst_x = (width - inst_size[0]) // 2
    inst_y = y_offset + 40
    cv2.putText(frame, instruction, (inst_x, inst_y), font, font_scale * 0.8, (200, 200, 200), thickness - 1)

    return frame


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
        published yet, or displays an error frame if the source encountered an error.
        """
        pts, time_base = await self.next_timestamp()

        # Check for error state first
        if self._slot.error:
            np_frame = create_error_frame(self._slot.error)
        else:
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

        frame = VideoFrame.from_ndarray(np_frame, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame
