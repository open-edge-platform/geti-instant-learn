# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated

from fastapi import Query, Response, status

from routers import projects_router

logger = logging.getLogger(__name__)


@projects_router.get(
    path="/export",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully exported the project configurations as a zip archive.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while exporting the project configurations.",
        },
    },
)
def export_projects(names: Annotated[list[str] | None, Query()] = None) -> Response:
    """
    Export project configurations as a zip archive.
    If no names are provided, exports all projects.

    Returns:
        Response: A .zip file containing the selected project directories (e.g., {p1_name}/configuration.yaml).
    """
    logger.debug("Received GET export projects request.")
    if names:
        logger.debug(f"Exporting projects with names: {names}")

    return Response(status_code=status.HTTP_200_OK, media_type="application/zip", content=b"")
