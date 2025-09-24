# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import Response, status

from routers import projects_router

logger = logging.getLogger(__name__)


@projects_router.get(
    path="/{project_id}/source",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved the source configuration for the project.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while retrieving the source configuration of the project.",
        },
    },
)
def get_source(project_id: UUID) -> Response:
    """
    Retrieve the source configuration of the project.
    """
    logger.debug(f"Received GET project {project_id} source request.")

    return Response(status_code=status.HTTP_200_OK, content={"project_source": {}})
