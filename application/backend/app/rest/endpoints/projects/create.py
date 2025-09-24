# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import HTTPException, Response, status

from db.repository.common import ResourceAlreadyExistsError
from db.repository.project import ProjectRepository
from dependencies import SessionDep
from rest.schemas.project import ProjectPostPayload
from routers import projects_router

logger = logging.getLogger(__name__)


@projects_router.post(
    path="",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {
            "description": "Successfully created a new project.",
        },
        status.HTTP_409_CONFLICT: {
            "description": "Project with this name already exists.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while creating a new project.",
        },
    },
)
def create_project(payload: ProjectPostPayload, db_session: SessionDep) -> Response:
    """Create a new project with the given name."""

    logger.debug(f"Attempting to create project with name: {payload.name}")
    repo = ProjectRepository(db_session)

    try:
        project = repo.create_project(name=payload.name, project_id=payload.id)
        logger.info(f"Successfully created {project.name} project with id {project.id}")
    except ResourceAlreadyExistsError as e:
        logger.error(f"Project creation failed: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during project creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create a project due to internal server error.",
        )

    location = f"/projects/{project.id}"
    return Response(status_code=status.HTTP_201_CREATED, headers={"Location": location})
