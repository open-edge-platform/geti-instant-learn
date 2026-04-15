# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MJPEG video streaming endpoint."""

from uuid import UUID

from starlette.responses import StreamingResponse

from api.routers import projects_router
from dependencies import MjpegStreamServiceDep, PipelineManagerDep
from runtime.services.mjpeg_stream import BOUNDARY
from runtime.visualizer import InferenceVisualizer


@projects_router.get(
    path="/{project_id}/stream",
    tags=["Stream"],
    responses={200: {"content": {"multipart/x-mixed-replace": {}}}},
    summary="MJPEG video stream of inference results",
)
async def stream_mjpeg(
    project_id: UUID,
    pipeline_manager: PipelineManagerDep,
    mjpeg_service: MjpegStreamServiceDep,
) -> StreamingResponse:
    """Stream annotated inference frames as MJPEG over HTTP."""
    output_slot = pipeline_manager.get_output_slot(project_id=project_id)
    visualizer = InferenceVisualizer(enable_visualization=True)

    def vis_info_provider():
        try:
            return pipeline_manager.get_visualization_info(project_id)
        except Exception:
            return None

    return StreamingResponse(
        mjpeg_service.stream(output_slot, visualizer, vis_info_provider),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
    )
