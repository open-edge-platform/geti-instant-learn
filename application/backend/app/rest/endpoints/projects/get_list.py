# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import TYPE_CHECKING

from fastapi import HTTPException, status

from db.repository.project import Project, ProjectRepository
from dependencies import SessionDep
from rest.schemas.project import ProjectListItem, ProjectsListSchema
from routers import projects_router

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


@projects_router.get(
    path="",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved a list of all available project configurations.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while retrieving available project configurations.",
        },
    },
)
def get_projects_list(db_session: SessionDep) -> ProjectsListSchema:
    """
    Retrieve a list of all available project configurations.
    """
    logger.debug("Received GET projects request.")
    repo = ProjectRepository(db_session)

    try:
        projects: Sequence[Project] = repo.get_all_projects()
        project_items = [ProjectListItem(id=project.id, name=project.name) for project in projects]
        logger.debug(f"Retrieved {len(project_items)} projects: {project_items}")

        return ProjectsListSchema(projects=project_items)

    except Exception:
        logger.exception("Unexpected error during retrieving projects list")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve projects due to internal server error.",
        )
