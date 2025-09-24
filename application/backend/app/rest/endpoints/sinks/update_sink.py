# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import Response, status

from routers import projects_router

logger = logging.getLogger(__name__)


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
