# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import Response, status

from routers import state_router

logger = logging.getLogger(__name__)


@state_router.get(
    path="",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved the name of the active pipeline.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while retrieving the active pipeline's name.",
        },
    },
)
def get_state() -> Response:
    """
    Retrieve the name of the currently active pipeline.
    """
    logger.debug("Received GET state request.")

    return Response(status_code=status.HTTP_200_OK)
