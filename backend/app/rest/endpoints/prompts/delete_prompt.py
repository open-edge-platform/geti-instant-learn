# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import Response, status

from routers import pipelines_router

logger = logging.getLogger(__name__)


@pipelines_router.delete(
    path="/{pipeline_id}/prompts/{prompt_id}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully deleted the prompt's directory and all its contents.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while deleting the prompt's directory.",
        },
    },
)
def delete_prompt(pipeline_id: UUID, prompt_id: UUID) -> Response:
    """
    Delete the prompt's directory and all its contents.
    """
    logger.debug(f"Received DELETE pipeline {pipeline_id} prompt {prompt_id} request.")

    return Response(status_code=status.HTTP_200_OK)
