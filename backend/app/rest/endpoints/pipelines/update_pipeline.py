# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import Response, status

from routers import pipelines_router

logger = logging.getLogger(__name__)


@pipelines_router.put(
    path="/{pipeline_id}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully updates the configuration for the pipeline.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while updating the configuration of the pipeline.",
        },
    },
)
def update_pipeline(pipeline_id: UUID) -> Response:
    """
    Update the pipeline's configuration.
    """
    logger.debug(f"Received PUT pipeline {pipeline_id} request.")

    return Response(status_code=status.HTTP_200_OK, content={"pipeline": {}})
