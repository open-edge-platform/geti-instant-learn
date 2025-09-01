# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import Response, status

from routers import state_router

logger = logging.getLogger(__name__)


@state_router.put(
    path="",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully updated the active pipeline's state.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while updating the state of the pipeline.",
        },
    },
)
def update_state() -> Response:
    """
    Sets the active pipeline by name.
    """
    logger.debug("Received PUT state request.")

    return Response(status_code=status.HTTP_200_OK)
