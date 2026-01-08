# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import HTTPException, Response, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.routers import projects_router
from dependencies import FrameServiceDep, PipelineManagerDep

logger = logging.getLogger(__name__)


class FrameCaptureResponse(BaseModel):
    """Response schema for frame capture endpoint."""

    frame_id: UUID


@projects_router.post(
    path="/{project_id}/frames",
    tags=["Frames"],
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {
            "description": "Frame captured successfully",
            "headers": {
                "Location": {
                    "description": "Relative URL to retrieve the captured frame",
                    "schema": {"type": "string"},
                    "example": "/projects/123e4567-e89b-12d3-a456-426614174000/"
                    "frames/550e8400-e29b-41d4-a716-446655440000",
                }
            },
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {"frame_id": {"type": "string"}},
                        "required": ["frame_id"],
                        "description": "Response schema for frame capture endpoint.",
                    },
                    "example": {"frame_id": "550e8400-e29b-41d4-a716-446655440000"},
                }
            },
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Project not found or no active source",
            "content": {
                "application/json": {
                    "examples": {
                        "project_missing": {
                            "summary": "Project not found",
                            "value": {
                                "detail": "Resource PROJECT with id 123e4567-e89b-12d3-a456-426614174000 not found"
                            },
                        },
                        "source_missing": {
                            "summary": "No active source",
                            "value": {
                                "detail": "Project 123e4567-e89b-12d3-a456-426614174000 has no active source. "
                                "Please connect a source before capturing frames."
                            },
                        },
                    }
                }
            },
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "Project is not active or frame capture failed",
            "content": {
                "application/json": {
                    "examples": {
                        "inactive": {
                            "summary": "Inactive project",
                            "value": {
                                "detail": "Cannot capture frame: project 123e4567-e89b-12d3-a456-426614174000 is "
                                "not active. Please activate the project before capturing frames."
                            },
                        },
                        "timeout": {
                            "summary": "Capture timeout",
                            "value": {"detail": "No frame received within 5.0 seconds. Pipeline may not be running."},
                        },
                        "generic_failure": {
                            "summary": "Other failure",
                            "value": {"detail": "Frame capture failed: internal processing error"},
                        },
                    }
                }
            },
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal server error",
            "content": {
                "application/json": {"example": {"detail": "An unexpected error occurred while capturing the frame"}}
            },
        },
    },
)
def capture_frame(project_id: UUID, pipeline_manager: PipelineManagerDep) -> Response:
    """
    Capture the latest frame from the video stream of the active project.
    Returns the frame ID in the response body and a Location header pointing to the captured frame.
    """
    frame_id = pipeline_manager.capture_frame(project_id)

    response = FrameCaptureResponse(frame_id=frame_id)

    return Response(
        status_code=status.HTTP_201_CREATED,
        headers={"Location": f"/projects/{project_id}/frames/{frame_id}"},
        content=response.model_dump_json(),
        media_type="application/json",
    )


@projects_router.get(
    path="/{project_id}/frames/{frame_id}",
    tags=["Frames"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Frame retrieved successfully",
            "content": {
                "image/jpeg": {
                    "examples": {
                        "sample": {
                            "summary": "Example JPEG (truncated)",
                            "value": "FFD8FFE000104A46494600010100000100010000FFDB...",
                        }
                    }
                }
            },
            "headers": {
                "Content-Type": {
                    "description": "MIME type of the returned frame",
                    "schema": {"type": "string"},
                    "example": "image/jpeg",
                }
            },
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Frame or project not found",
            "content": {"application/json": {"example": {"detail": "Frame not found"}}},
        },
    },
)
def get_frame(project_id: UUID, frame_id: UUID, frame_service: FrameServiceDep) -> Response:
    """
    Retrieve a captured frame as JPEG.
    """
    frame_path = frame_service.get_frame_path(project_id, frame_id)
    if frame_path is None or not frame_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Frame not found")

    return FileResponse(frame_path, media_type="image/jpeg")
