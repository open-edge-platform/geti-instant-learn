# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import Response, status

from dependencies import SourceServiceDep
from routers import projects_router
from services.schemas.source import SourceCreateSchema, SourceSchema, SourcesListSchema, SourceUpdateSchema

logger = logging.getLogger(__name__)


@projects_router.get(
    path="/{project_id}/sources",
    tags=["Sources"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successfully retrieved the sources configuration for the project."},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Unexpected error occurred."},
    },
)
def get_sources(project_id: UUID, source_service: SourceServiceDep) -> SourcesListSchema:
    """
    Retrieve the source configuration of the project.
    """
    return source_service.list_sources(project_id)


@projects_router.post(
    path="/{project_id}/sources",
    tags=["Sources"],
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Source created."},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found."},
        status.HTTP_409_CONFLICT: {"description": "Source of this type already exists in project."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Unexpected error occurred."},
    },
)
def create_source(
    project_id: UUID,
    payload: SourceCreateSchema,
    source_service: SourceServiceDep,
) -> SourceSchema:
    """
    Create a new source configuration for the project.
    """
    return source_service.create_source(project_id=project_id, create_data=payload)


@projects_router.put(
    path="/{project_id}/sources/{source_id}",
    tags=["Sources"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successfully updated the configuration for the project's source."},
        status.HTTP_404_NOT_FOUND: {"description": "Project or source not found."},
        status.HTTP_409_CONFLICT: {"description": "Source type change is not allowed."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Unexpected error occurred."},
    },
)
def update_source(
    project_id: UUID,
    source_id: UUID,
    payload: SourceUpdateSchema,
    source_service: SourceServiceDep,
) -> SourceSchema:
    """
    Update the project's source configuration.
    """
    return source_service.update_source(project_id=project_id, source_id=source_id, update_data=payload)


@projects_router.delete(
    path="/{project_id}/sources/{source_id}",
    tags=["Sources"],
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Successfully deleted the project's source configuration.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while deleting the project's source configuration.",
        },
    },
)
def delete_source(
    project_id: UUID,
    source_id: UUID,
    source_service: SourceServiceDep,
) -> Response:
    """
    Delete the specified project's source configuration.
    """
    source_service.delete_source(project_id=project_id, source_id=source_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
