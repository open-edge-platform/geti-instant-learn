# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import Response, status

from routers import pipelines_router

logger = logging.getLogger(__name__)


@pipelines_router.get(
    path="",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved a list of all available pipeline configurations.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while retrieving available pipeline configurations.",
        },
    },
)
def get_pipelines_list() -> Response:
    """
    Retrieve a list of all available pipeline configurations.
    """
    logger.debug("Received GET pipelines request.")

    return Response(status_code=status.HTTP_200_OK, content="pipelinesssss")
