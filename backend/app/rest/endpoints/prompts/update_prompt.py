# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import Response, status

from routers import projects_router

logger = logging.getLogger(__name__)


@projects_router.put(
    path="/{project_id}/prompts/{prompt_id}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully updated the files for the prompt.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while updating the prompt details.",
        },
    },
)
def update_prompt(project_id: UUID, prompt_id: UUID) -> Response:
    """
    Update the existing files of the prompt.
    """
    logger.debug(f"Received PUT project {project_id} prompt {prompt_id} request.")

    return Response(status_code=status.HTTP_200_OK)
