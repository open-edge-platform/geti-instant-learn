# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import HTTPException, status

from db.repository.common import ResourceNotFoundError
from db.repository.project import ProjectRepository
from dependencies import SessionDep
from rest.schemas.processor import ProcessorSchema
from rest.schemas.project import ProjectSchema
from rest.schemas.sink import SinkSchema
from rest.schemas.source import SourceSchema
from routers import projects_router

logger = logging.getLogger(__name__)


@projects_router.get(
    path="/{project_id}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved the configuration for a project.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while retrieving the configuration of a project.",
        },
    },
)
def get_project(project_id: UUID, db_session: SessionDep) -> ProjectSchema:
    """
    Retrieve the project's configuration.
    """
    logger.debug(f"Received GET project {project_id} request.")

    repo = ProjectRepository(db_session)

    try:
        project = repo.get_project_by_id(project_id)
        project_response = ProjectSchema(id=project.id, name=project.name)
        if project.source:
            project_response.source = SourceSchema(
                id=project.source.id,
                type=project.source.type,
                config=project.source.config,
            )

        if project.processor:
            project_response.processor = ProcessorSchema(
                id=project.processor.id,
                type=project.processor.type,
                config=project.processor.config,
                name=project.processor.name,
            )

        if project.sink:
            project_response.sink = SinkSchema(
                id=project.sink.id,
                config=project.sink.config,
            )

        return project_response

    except ResourceNotFoundError:
        logger.exception(f"Project with id {project_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id {project_id} not found.",
        )

    except Exception as e:
        logger.exception(f"Unexpected error during retrieval of a project with id {project_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve the project due to internal server error.",
        )
