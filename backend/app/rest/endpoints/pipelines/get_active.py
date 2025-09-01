# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import Response, status

from routers import pipelines_router

logger = logging.getLogger(__name__)


@pipelines_router.get(
    path="/active",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved the configuration of the currently active pipeline.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while retrieving the active pipeline configuration.",
        },
    },
)
def get_active_pipeline() -> Response:
    """
    Retrieve the configuration of the currently active pipeline.
    """
    logger.debug("Received GET active pipeline request.")

    return Response(status_code=status.HTTP_200_OK)
