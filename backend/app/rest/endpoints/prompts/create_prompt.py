# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated
from uuid import UUID

from fastapi import Form, Response, status

from routers import pipelines_router

logger = logging.getLogger(__name__)


@pipelines_router.post(
    path="/{pipeline_id}/prompts",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {
            "description": "Successfully added a new prompt to the pipeline.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while adding a new prompt to the pipeline.",
        },
    },
)
def create_prompt(
    pipeline_id: UUID,
    prompt_name: Annotated[str, Form()],
    prompt_type: Annotated[str, Form()],
    # image: Annotated[UploadFile, File()],
    # annotation_data: Annotated[UploadFile, File()]
) -> Response:
    """
    Add a new prompt to the pipeline.
    """
    logger.debug(f"Received POST pipeline {pipeline_id} prompt request, name: {prompt_name}, type: {prompt_type}")

    return Response(status_code=status.HTTP_201_CREATED)
