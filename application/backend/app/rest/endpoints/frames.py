# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Response, status
from fastapi.responses import FileResponse

from dependencies import get_frame_service
from routers import projects_router
from services.errors import ResourceNotFoundError, ServiceError
from services.frame import FrameService

logger = logging.getLogger(__name__)


@projects_router.post(
    path="/{project_id}/frames:capture",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Frame captured successfully"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found or no connected source"},
        status.HTTP_400_BAD_REQUEST: {"description": "Project is not active or frame capture failed"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)
def capture_frame(project_id: UUID, frame_service: Annotated[FrameService, Depends(get_frame_service)]) -> Response:
    """
    Capture the latest frame from the video stream of the active project.
    Returns a response with a Location header pointing to the captured frame.
    """
    logger.debug(f"Received POST capture frame for project {project_id} request.")

    try:
        frame_id = frame_service.capture_frame(project_id)
        return Response(
            status_code=status.HTTP_201_CREATED, headers={"Location": f"/projects/{project_id}/frames/{frame_id}"}
        )
    except ResourceNotFoundError as e:
        logger.warning(f"Resource not found during frame capture: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ServiceError as e:
        logger.warning(f"Service error during frame capture: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error capturing frame: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while capturing the frame",
        )


@projects_router.get(
    path="/{project_id}/frames/{frame_id}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Frame retrieved successfully", "content": {"image/jpeg": {}}},
        status.HTTP_404_NOT_FOUND: {"description": "Frame or project not found"},
    },
)
def get_frame(
    project_id: UUID, frame_id: UUID, frame_service: Annotated[FrameService, Depends(get_frame_service)]
) -> Response:
    """
    Retrieve a captured frame as JPEG.
    """
    logger.debug(f"Received GET project {project_id} frame {frame_id} request.")

    frame_path = frame_service.get_frame_path(project_id, frame_id)
    if frame_path is None or not frame_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Frame not found")

    return FileResponse(frame_path, media_type="image/jpeg")
