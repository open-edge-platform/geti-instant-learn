# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import Response, status

from routers import pipelines_router

logger = logging.getLogger(__name__)


@pipelines_router.delete(
    path="/{pipeline_id}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully deleted the pipeline.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while deleting the pipeline.",
        },
    },
)
def delete_pipeline(pipeline_id: UUID) -> Response:
    """
    Delete the specified pipeline.
    """
    logger.debug(f"Received DELETE pipeline {pipeline_id} request.")

    return Response(status_code=status.HTTP_200_OK, content="Pipeline deleted successfully.")
