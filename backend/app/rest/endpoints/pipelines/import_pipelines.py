# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import Response, status

from routers import pipelines_router

logger = logging.getLogger(__name__)


@pipelines_router.post(
    path="/import",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {
            "description": "Successfully imported a new pipeline from an archive.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while importing the pipeline.",
        },
    },
)
def import_pipelines() -> Response:
    """
    Import pipelines from a .zip archive.
    The server will copy the pipeline configurations into the application's configuration directory.
    If a pipeline with the same name already exists, the import for that specific pipeline
    will be rejected with an error to prevent accidental overwrites.
    """
    logger.debug("Received POST import pipeline request.")

    return Response(status_code=status.HTTP_201_CREATED)
