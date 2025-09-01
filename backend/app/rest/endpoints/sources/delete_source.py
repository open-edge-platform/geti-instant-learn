# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import Response, status

from routers import pipelines_router

logger = logging.getLogger(__name__)


@pipelines_router.delete(
    path="/{pipeline_id}/source",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully deleted the pipeline's source configuration.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while deleting the pipeline's source configuration.",
        },
    },
)
def delete_source(pipeline_id: UUID) -> Response:
    """
    Delete the specified pipeline's source configuration.
    """
    logger.debug(f"Received DELETE pipeline {pipeline_id} source request.")

    return Response(
        status_code=status.HTTP_200_OK, content=f"Source for the pipeline {pipeline_id} deleted successfully."
    )
