# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import Response, status

from routers import projects_router

logger = logging.getLogger(__name__)


@projects_router.get(
    path="/{project_id}/prompts",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved the list of all prompts for the project.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while retrieving the list of prompts for the project.",
        },
    },
)
def get_all_prompts(project_id: UUID) -> Response:
    """
    Retrieve a list of all prompts for the project.
    """
    logger.debug(f"Received GET project {project_id} prompts request.")

    return Response(status_code=status.HTTP_200_OK, content={"project_prompts": []})
