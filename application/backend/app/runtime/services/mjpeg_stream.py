# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MJPEG streaming service."""

import asyncio
import contextlib
import itertools
import logging
from collections.abc import AsyncGenerator, Callable

import cv2
import numpy as np

from domain.services.schemas.label import VisualizationInfo
from domain.services.schemas.processor import OutputData
from runtime.core.components.broadcaster import FrameSlot
from runtime.visualizer import InferenceVisualizer

logger = logging.getLogger(__name__)

BOUNDARY = "frame"
TIMING_LOG_INTERVAL_FRAMES = 30
_STREAM_ID_COUNTER = itertools.count(1)


def _encode_jpeg(bgr: np.ndarray, quality: int) -> bytes:
    """Encode a BGR frame to JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


class MjpegStreamService:
    """Produces an MJPEG multipart byte stream from inference output frames."""

    def __init__(self, quality: int, max_fps: int) -> None:
        self._quality = quality
        self._max_fps = max_fps

    async def stream(
        self,
        output_slot: FrameSlot[OutputData],
        visualizer: InferenceVisualizer,
        vis_info_provider: Callable[[], VisualizationInfo | None],
    ) -> AsyncGenerator[bytes, None]:
        """Async generator that yields MJPEG multipart frames."""
        min_interval = 1.0 / self._max_fps
        last_output: OutputData | None = None
        last_yield_time = 0.0
        boundary = BOUNDARY.encode()
        loop = asyncio.get_event_loop()
        stream_id = next(_STREAM_ID_COUNTER)
        stream_start_time = loop.time()
        emitted_frames = 0
        no_frame_polls = 0
        duplicate_frame_polls = 0
        throttled_polls = 0
        vis_info_time = 0.0
        visualize_time = 0.0
        color_convert_time = 0.0
        encode_time = 0.0
        multipart_build_time = 0.0

        def log_timing_summary(reason: str) -> None:
            active_time = max(loop.time() - stream_start_time, 0.0)
            average_frame_time = (
                (vis_info_time + visualize_time + color_convert_time + encode_time + multipart_build_time)
                / emitted_frames
                if emitted_frames
                else 0.0
            )
            logger.info(
                (
                    "MJPEG stream %s timing (%s): frames=%d active_s=%.3f avg_frame_ms=%.2f "
                    "vis_info_ms=%.2f visualize_ms=%.2f color_ms=%.2f encode_ms=%.2f multipart_ms=%.2f "
                    "polls_no_frame=%d polls_duplicate=%d polls_throttled=%d"
                ),
                stream_id,
                reason,
                emitted_frames,
                active_time,
                average_frame_time * 1000,
                (vis_info_time / emitted_frames) * 1000 if emitted_frames else 0.0,
                (visualize_time / emitted_frames) * 1000 if emitted_frames else 0.0,
                (color_convert_time / emitted_frames) * 1000 if emitted_frames else 0.0,
                (encode_time / emitted_frames) * 1000 if emitted_frames else 0.0,
                (multipart_build_time / emitted_frames) * 1000 if emitted_frames else 0.0,
                no_frame_polls,
                duplicate_frame_polls,
                throttled_polls,
            )

        try:
            while True:
                output_data = output_slot.latest
                if output_data is None:
                    no_frame_polls += 1
                    await asyncio.sleep(0.001)
                    continue
                if output_data is last_output:
                    duplicate_frame_polls += 1
                    await asyncio.sleep(0.001)
                    continue

                now = loop.time()
                if now - last_yield_time < min_interval:
                    throttled_polls += 1
                    await asyncio.sleep(0.001)
                    continue

                last_output = output_data

                vis_info_start = loop.time()
                vis_info = vis_info_provider()
                vis_info_time += loop.time() - vis_info_start

                visualize_start = loop.time()
                rgb = visualizer.visualize(output_data=output_data, visualization_info=vis_info)
                visualize_time += loop.time() - visualize_start

                color_convert_start = loop.time()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                color_convert_time += loop.time() - color_convert_start

                encode_start = loop.time()
                jpeg = await loop.run_in_executor(None, _encode_jpeg, bgr, self._quality)
                encode_time += loop.time() - encode_start

                multipart_start = loop.time()
                multipart_chunk = (
                    b"--" + boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                    b"\r\n" + jpeg + b"\r\n"
                )
                multipart_build_time += loop.time() - multipart_start

                emitted_frames += 1
                last_yield_time = loop.time()

                if emitted_frames % TIMING_LOG_INTERVAL_FRAMES == 0:
                    log_timing_summary(reason="interval")

                yield multipart_chunk
        finally:
            with contextlib.suppress(RuntimeError):
                log_timing_summary(reason="closed")

