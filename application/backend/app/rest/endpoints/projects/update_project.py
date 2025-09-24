# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import HTTPException, status

from db.repository.common import ResourceNotFoundError
from db.repository.project import ProjectRepository
from dependencies import SessionDep
from rest.schemas.project import ProjectPutPayload, ProjectSchema
from routers import projects_router

logger = logging.getLogger(__name__)


@projects_router.put(
    path="/{project_id}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully updates the configuration for the project.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while updating the configuration of the project.",
        },
    },
)
def update_project(project_id: UUID, payload: ProjectPutPayload, db_session: SessionDep) -> ProjectSchema:
    """
    Update the project's configuration.
    """
    logger.debug(f"Received PUT project {project_id} request.")
    repo = ProjectRepository(db_session)
    try:
        updated_project = repo.update_project(project_id=project_id, new_name=payload.name)
        logger.info(f"Successfully updated project with id {updated_project.id}, new name: {updated_project.name}")

        return ProjectSchema(id=updated_project.id, name=updated_project.name)

    except ResourceNotFoundError as e:
        logger.exception(f"Project update failed: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during project update: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update the project due to internal server error.",
        )
