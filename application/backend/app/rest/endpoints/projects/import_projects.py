# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import Response, status

from routers import projects_router

logger = logging.getLogger(__name__)


@projects_router.post(
    path="/import",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {
            "description": "Successfully imported a new project from an archive.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while importing the project.",
        },
    },
)
def import_projects() -> Response:
    """
    Import projects from a .zip archive.
    The server will copy the project configurations into the application's configuration directory.
    If a project with the same name already exists, the import for that specific project
    will be rejected with an error to prevent accidental overwrites.
    """
    logger.debug("Received POST import project request.")

    return Response(status_code=status.HTTP_201_CREATED)
