# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import Response, status

from routers import pipelines_router

logger = logging.getLogger(__name__)


@pipelines_router.put(
    path="/{pipeline_id}/sink",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_201_CREATED: {
            "description": "Successfully created the configuration for the pipeline's sink.",
        },
        status.HTTP_200_OK: {
            "description": "Successfully updates the configuration for the pipeline's sink.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while updating the configuration of the pipeline's sink.",
        },
    },
)
def update_sink(pipeline_id: UUID) -> Response:
    """
    Update the pipeline's configuration.
    """
    logger.debug(f"Received PUT pipeline {pipeline_id} sink request.")

    return Response(status_code=status.HTTP_200_OK, content={"pipeline_sink": {}})
