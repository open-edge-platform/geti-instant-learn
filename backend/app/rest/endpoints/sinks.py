# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import Response, status

from routers import projects_router

logger = logging.getLogger(__name__)


@projects_router.get(
    path="/{project_id}/sink",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved the sink configuration for the project.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while retrieving the sink configuration of the project.",
        },
    },
)
def get_sink(project_id: UUID) -> Response:
    """
    Retrieve the sink configuration of the project.
    """
    logger.debug(f"Received GET project {project_id} sink request.")
    return Response(status_code=status.HTTP_200_OK, content={"project_sink": {}})


@projects_router.put(
    path="/{project_id}/sink",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_201_CREATED: {
            "description": "Successfully created the configuration for the project's sink.",
        },
        status.HTTP_200_OK: {
            "description": "Successfully updates the configuration for the project's sink.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while updating the configuration of the project's sink.",
        },
    },
)
def update_sink(project_id: UUID) -> Response:
    """
    Update the project's configuration.
    """
    logger.debug(f"Received PUT project {project_id} sink request.")
    return Response(status_code=status.HTTP_200_OK, content={"project_sink": {}})


@projects_router.delete(
    path="/{project_id}/sink",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully deleted the project's sink configuration.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while deleting the project's sink configuration.",
        },
    },
)
def delete_sink(project_id: UUID) -> Response:
    """
    Delete the specified project's sink configuration.
    """
    logger.debug(f"Received DELETE project {project_id} sink request.")
    return Response(status_code=status.HTTP_200_OK, content=f"Sink for the project {project_id} deleted successfully.")
