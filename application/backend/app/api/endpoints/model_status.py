# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Project-scoped endpoints exposing the model (processor) loading state.

- ``GET /api/v1/projects/{project_id}/model-status`` returns a one-shot snapshot.
- ``GET /api/v1/projects/{project_id}/model-status/stream`` opens a Server-Sent
  Events stream that pushes status changes as they happen.

The PipelineManager broadcasts a single global model status (only one pipeline
can be active at a time), so the SSE generator filters events to the requested
project. When another project is active or no pipeline is running, the stream
still reports IDLE so the UI can recover when the project gets activated again.
"""

import logging
from collections.abc import AsyncIterable
from uuid import UUID

from fastapi import status
from fastapi.sse import EventSourceResponse

from api.routers import projects_router
from dependencies import PipelineManagerDep, ProjectServiceDep
from domain.services.schemas.model_status import ModelStatusSchema
from runtime.pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)


def _scoped_snapshot(snapshot: ModelStatusSchema, project_id: UUID) -> ModelStatusSchema:
    """Return the snapshot if it belongs to the requested project, else IDLE."""
    if snapshot.project_id == project_id:
        return snapshot
    return ModelStatusSchema.idle(project_id=project_id)


async def _model_status_stream(pipeline_manager: PipelineManager, project_id: UUID) -> AsyncIterable[ModelStatusSchema]:
    """Yield project-scoped model status snapshots.

    FastAPI's ``EventSourceResponse`` JSON-encodes each yielded Pydantic model
    into the ``data:`` field of an SSE event and handles keep-alive pings,
    ``Cache-Control``/``X-Accel-Buffering`` headers and client-disconnect
    cleanup automatically.
    """
    queue = pipeline_manager.subscribe_status()
    try:
        # Initial snapshot so the client immediately knows the current state.
        yield _scoped_snapshot(pipeline_manager.get_status(), project_id)
        while True:
            snapshot = await queue.get()
            yield _scoped_snapshot(snapshot, project_id)
    finally:
        pipeline_manager.unsubscribe_status(queue)


@projects_router.get(
    path="/{project_id}/model-status",
    tags=["Model Status"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Current model status snapshot for the project."},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found."},
    },
)
def get_model_status(
    project_id: UUID, project_service: ProjectServiceDep, pipeline_manager: PipelineManagerDep
) -> ModelStatusSchema:
    """Return a one-shot snapshot of the current model loading status."""
    project_service.get_project(project_id)  # 404 if missing
    return _scoped_snapshot(pipeline_manager.get_status(), project_id)


@projects_router.get(
    path="/{project_id}/model-status/stream",
    tags=["Model Status"],
    response_class=EventSourceResponse,
    responses={
        status.HTTP_200_OK: {
            "description": "Server-Sent Events stream of model status updates.",
            "content": {"text/event-stream": {}},
        },
        status.HTTP_404_NOT_FOUND: {"description": "Project not found."},
    },
)
async def stream_model_status(
    project_id: UUID, project_service: ProjectServiceDep, pipeline_manager: PipelineManagerDep
) -> AsyncIterable[ModelStatusSchema]:
    """Open an SSE stream of model status updates for the project.

    The stream emits an initial snapshot, then a snapshot per state transition
    (loading reference batch, loading model, ready, error, idle).
    """
    project_service.get_project(project_id)  # 404 if missing
    async for snapshot in _model_status_stream(pipeline_manager, project_id):
        yield snapshot
