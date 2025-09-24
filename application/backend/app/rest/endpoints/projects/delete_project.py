# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import HTTPException, Response, status

from db.repository.common import ResourceNotFoundError
from db.repository.project import ProjectRepository
from dependencies import SessionDep
from routers import projects_router

logger = logging.getLogger(__name__)


@projects_router.delete(
    path="/{project_id}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Successfully deleted the project.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while deleting the project.",
        },
    },
)
def delete_project(project_id: UUID, db_session: SessionDep) -> Response:
    """
    Delete the specified project.
    """
    logger.debug(f"Received DELETE project {project_id} request.")
    repo = ProjectRepository(db_session)
    try:
        repo.delete_project(project_id)
    except ResourceNotFoundError:
        logger.warning(f"Project with id {project_id} not found during delete operation.")
    except Exception as e:
        logger.exception(f"Error occurred while deleting project with id {project_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete project with id {project_id} due to an internal error.",
        )

    logger.info(f"Successfully deleted project with id {project_id}.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
