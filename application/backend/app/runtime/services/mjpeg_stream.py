# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MJPEG streaming service."""

import asyncio
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

        while True:
            output_data = output_slot.latest
            if output_data is None or output_data is last_output:
                await asyncio.sleep(0.001)
                continue

            now = loop.time()
            if now - last_yield_time < min_interval:
                await asyncio.sleep(0.001)
                continue

            last_output = output_data
            vis_info = vis_info_provider()

            rgb = visualizer.visualize(output_data=output_data, visualization_info=vis_info)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            jpeg = await loop.run_in_executor(None, _encode_jpeg, bgr, self._quality)

            yield (
                b"--" + boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                b"\r\n" + jpeg + b"\r\n"
            )

            last_yield_time = loop.time()
