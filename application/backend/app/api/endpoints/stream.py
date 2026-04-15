# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MJPEG video streaming endpoint."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Annotated
from uuid import UUID

import cv2
from fastapi import Depends, Query
from starlette.responses import StreamingResponse

from api.routers import projects_router
from dependencies import get_pipeline_manager
from domain.services.schemas.label import VisualizationInfo
from domain.services.schemas.processor import OutputData
from runtime.core.components.broadcaster import FrameSlot
from runtime.pipeline_manager import PipelineManager
from runtime.visualizer import InferenceVisualizer
from settings import get_settings

logger = logging.getLogger(__name__)

BOUNDARY = "frame"


def _visualize_and_encode(
    output_data: OutputData,
    visualizer: InferenceVisualizer,
    vis_info: VisualizationInfo | None,
    quality: int,
) -> bytes:
    """CPU-bound: visualize predictions and encode to JPEG."""
    np_frame = visualizer.visualize(output_data=output_data, visualization_info=vis_info)
    bgr = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


async def _mjpeg_frames(
    output_slot: FrameSlot[OutputData],
    visualizer: InferenceVisualizer,
    vis_info_provider: callable,
    quality: int,
    max_fps: int,
) -> AsyncGenerator[bytes, None]:
    """Async generator that yields MJPEG multipart frames."""
    min_interval = 1.0 / max_fps
    last_output: OutputData | None = None
    last_yield_time = 0.0
    boundary = BOUNDARY.encode()
    loop = asyncio.get_event_loop()

    while True:
        # Poll until a new frame arrives from the pipeline
        output_data = output_slot.latest
        if output_data is None or output_data is last_output:
            await asyncio.sleep(0.001)
            continue

        # Throttle: skip if not enough time since last yield
        now = loop.time()
        if now - last_yield_time < min_interval:
            await asyncio.sleep(0.001)
            continue

        last_output = output_data
        vis_info: VisualizationInfo | None = vis_info_provider()

        # Run CPU-heavy visualization + encoding off the event loop
        jpeg = await loop.run_in_executor(None, _visualize_and_encode, output_data, visualizer, vis_info, quality)

        yield (
            b"--" + boundary + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
            b"\r\n" + jpeg + b"\r\n"
        )

        last_yield_time = loop.time()


@projects_router.get(
    path="/{project_id}/stream",
    tags=["Stream"],
    responses={200: {"content": {"multipart/x-mixed-replace": {}}}},
    summary="MJPEG video stream of inference results",
)
async def stream_mjpeg(
    project_id: UUID,
    pipeline_manager: Annotated[PipelineManager, Depends(get_pipeline_manager)],
    quality: int = Query(default=None, ge=1, le=100, description="JPEG quality (1-100)"),
    fps: int = Query(default=None, ge=1, le=60, description="Maximum frames per second"),
) -> StreamingResponse:
    """Stream annotated inference frames as MJPEG over HTTP."""
    settings = get_settings()
    quality = quality if quality is not None else settings.mjpeg_quality
    fps = fps if fps is not None else settings.mjpeg_max_fps

    output_slot = pipeline_manager.get_output_slot(project_id=project_id)
    visualizer = InferenceVisualizer(enable_visualization=True)

    def vis_info_provider() -> VisualizationInfo | None:
        try:
            return pipeline_manager.get_visualization_info(project_id)
        except Exception:
            return None

    return StreamingResponse(
        _mjpeg_frames(output_slot, visualizer, vis_info_provider, quality, fps),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
    )
