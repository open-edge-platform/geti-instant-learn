# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import HTTPException, status

from db.repository.common import ResourceNotFoundError, ResourceType
from db.repository.project import ProjectRepository
from dependencies import SessionDep
from rest.schemas.processor import ProcessorSchema
from rest.schemas.project import ProjectSchema
from rest.schemas.sink import SinkSchema
from rest.schemas.source import SourceSchema
from routers import projects_router

logger = logging.getLogger(__name__)


@projects_router.get(
    path="/active",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved the configuration of the currently active project.",
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "No active project found.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while retrieving the active project configuration.",
        },
    },
)
def get_active_project(db_session: SessionDep) -> ProjectSchema:
    """
    Retrieve the configuration of the currently active project.
    """
    logger.debug("Received GET active project request.")
    repo = ProjectRepository(db_session)

    try:
        active_project = repo.get_active_project()
        if not active_project:
            raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, message="No active project found.")

        logger.info(f"Active project retrieved: {active_project.name} with id {active_project.id}")

        project_response = ProjectSchema(id=active_project.id, name=active_project.name)
        if active_project.source:
            project_response.source = SourceSchema(
                id=active_project.source.id,
                type=active_project.source.type,
                config=active_project.source.config,
            )

        if active_project.processor:
            project_response.processor = ProcessorSchema(
                id=active_project.processor.id,
                type=active_project.processor.type,
                config=active_project.processor.config,
                name=active_project.processor.name,
            )

        if active_project.sink:
            project_response.sink = SinkSchema(
                id=active_project.sink.id,
                config=active_project.sink.config,
            )

        return project_response

    except ResourceNotFoundError:
        logger.exception("No active project found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active project found.",
        )
    except Exception as e:
        logger.exception(f"Unexpected error during retrieval of active project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve the active project due to internal server error.",
        )
