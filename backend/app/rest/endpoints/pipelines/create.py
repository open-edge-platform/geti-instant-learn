# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import Response, status

from dependencies import SessionDep
from routers import pipelines_router

logger = logging.getLogger(__name__)


@pipelines_router.post(
    path="",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {
            "description": "Successfully created a new pipeline.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while creating a new pipeline.",
        },
    },
)
def create_pipeline(db_session: SessionDep) -> Response:
    """
    Create a new pipeline configuration.
    """
    logger.debug("Received POST pipelines request.")
    logger.debug(f"db conneection: {db_session}")

    return Response(status_code=status.HTTP_201_CREATED)
