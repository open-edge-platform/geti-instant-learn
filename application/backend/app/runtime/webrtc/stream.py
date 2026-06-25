# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import textwrap
import time
from collections.abc import Callable

import cv2
import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame

from domain.services.schemas.label import VisualizationInfo
from domain.services.schemas.processor import ErrorData, OutputData
from runtime.core.components.broadcaster import FrameSlot
from runtime.webrtc.visualizer import InferenceVisualizer
from settings import get_settings

logger = logging.getLogger(__name__)

FALLBACK_FRAME = np.full((64, 64, 3), 16, dtype=np.uint8)


def create_error_frame(error_data: ErrorData, width: int = 1280, height: int = 720) -> np.ndarray:
    """Create a frame with error text overlay.

    Args:
        error_data: The error data containing the message and component.
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
    title = f"{error_data.component.name.capitalize()} Error"
    title_size = cv2.getTextSize(title, font, font_scale * 1.2, thickness + 1)[0]
    title_x = (width - title_size[0]) // 2
    title_y = height // 3
    cv2.putText(frame, title, (title_x, title_y), font, font_scale * 1.2, (255, 100, 100), thickness + 1)

    # Wrap and display error message
    wrapped_lines = textwrap.wrap(error_data.message, width=60)
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
        output_slot: FrameSlot[OutputData | ErrorData],
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
        settings = get_settings()
        self._stats_enabled = settings.enable_cpu_monitoring
        self._stats_interval_secs = settings.cpu_monitoring_interval_secs
        self._stats_started_at_s = time.perf_counter()
        self._recv_count = 0
        self._new_frame_count = 0
        self._cached_frame_count = 0
        self._fallback_frame_count = 0
        self._error_frame_count = 0
        self._recv_wall_time_s = 0.0
        self._recv_cpu_time_s = 0.0

    async def recv(self) -> VideoFrame:
        """Return the next video frame for WebRTC streaming.

        Reads ``self._slot.latest`` on every call.  When a new
        ``OutputData`` is detected (identity check), visualization and
        tracing are applied and the rendered numpy array is cached.
        Subsequent calls that see the same ``OutputData`` reuse the cache.

        Falls back to a small dark-gray placeholder when no frame has been
        published yet, or displays an error frame if the source encountered an error.
        """
        wall_started_at_s = time.perf_counter()
        cpu_started_at_s = time.thread_time()
        frame_kind = "fallback"

        pts, time_base = await self.next_timestamp()

        # Check for error state first
        output_data = self._slot.latest
        if isinstance(output_data, ErrorData):
            frame_kind = "error"
            np_frame = create_error_frame(output_data)
        elif output_data is not None and output_data is not self._last_output:
            frame_kind = "new"
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
                logger.debug(output_data.trace.format_log())
        else:
            # Use cached frame or fallback only when no new output
            frame_kind = "cached" if self._last_frame is not None else "fallback"
            np_frame = self._last_frame if self._last_frame is not None else FALLBACK_FRAME

        frame = VideoFrame.from_ndarray(np_frame, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        self._record_recv_stats(
            frame_kind=frame_kind,
            wall_time_s=time.perf_counter() - wall_started_at_s,
            cpu_time_s=time.thread_time() - cpu_started_at_s,
        )
        return frame

    def _record_recv_stats(self, frame_kind: str, wall_time_s: float, cpu_time_s: float) -> None:
        if not self._stats_enabled:
            return

        self._recv_count += 1
        self._recv_wall_time_s += wall_time_s
        self._recv_cpu_time_s += cpu_time_s
        match frame_kind:
            case "new":
                self._new_frame_count += 1
            case "cached":
                self._cached_frame_count += 1
            case "error":
                self._error_frame_count += 1
            case _:
                self._fallback_frame_count += 1

        now_s = time.perf_counter()
        elapsed_s = now_s - self._stats_started_at_s
        if elapsed_s < self._stats_interval_secs:
            return

        recv_rate = self._recv_count / elapsed_s if elapsed_s > 0 else 0.0
        avg_wall_ms = (self._recv_wall_time_s / self._recv_count) * 1000 if self._recv_count else 0.0
        avg_cpu_ms = (self._recv_cpu_time_s / self._recv_count) * 1000 if self._recv_count else 0.0
        logger.info(
            "webrtc_recv_stats interval_secs=%.1f recv_count=%d recv_rate=%.1f new_frames=%d cached_frames=%d "
            "fallback_frames=%d error_frames=%d avg_recv_wall_ms=%.3f avg_recv_cpu_ms=%.3f",
            elapsed_s,
            self._recv_count,
            recv_rate,
            self._new_frame_count,
            self._cached_frame_count,
            self._fallback_frame_count,
            self._error_frame_count,
            avg_wall_ms,
            avg_cpu_ms,
        )
        self._reset_recv_stats(now_s)

    def _reset_recv_stats(self, started_at_s: float) -> None:
        self._stats_started_at_s = started_at_s
        self._recv_count = 0
        self._new_frame_count = 0
        self._cached_frame_count = 0
        self._fallback_frame_count = 0
        self._error_frame_count = 0
        self._recv_wall_time_s = 0.0
        self._recv_cpu_time_s = 0.0
