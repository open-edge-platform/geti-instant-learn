# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated
from uuid import UUID

from fastapi import Query, Response, status

from dependencies import ProjectServiceDep
from routers import projects_router
from services.schemas.project import (
    ProjectCreateSchema,
    ProjectSchema,
    ProjectsListSchema,
    ProjectUpdateSchema,
)

logger = logging.getLogger(__name__)


@projects_router.post(
    path="",
    tags=["Projects"],
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {
            "description": "Successfully created a new project.",
            "headers": {
                "Location": {
                    "description": "Relative URL to retrieve the created project",
                    "schema": {"type": "string"},
                    "example": "/projects/123e4567-e89b-12d3-a456-426614174000",
                }
            },
            "content": {
                "application/json": {
                    "example": {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "name": "My Project",
                        "active": True,
                    }
                }
            },
        },
        status.HTTP_409_CONFLICT: {
            "description": "Project with this name already exists.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while creating a new project.",
        },
    },
)
def create_project(
    payload: ProjectCreateSchema,
    project_service: ProjectServiceDep,
) -> Response:
    """Create a new project with the given name."""
    project = project_service.create_project(payload)
    logger.info(f"Successfully created '{project.name}' project with id {project.id}")

    return Response(
        status_code=status.HTTP_201_CREATED,
        headers={"Location": f"/projects/{project.id}"},
        content=project.model_dump_json(),
        media_type="application/json",
    )


@projects_router.delete(
    path="/{project_id}",
    tags=["Projects"],
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Successfully deleted the project.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while deleting the project.",
        },
    },
)
def delete_project(
    project_id: UUID,
    project_service: ProjectServiceDep,
) -> Response:
    """Delete the specified project."""
    project_service.delete_project(project_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@projects_router.get(
    path="/active",
    tags=["Projects"],
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
def get_active_project(
    project_service: ProjectServiceDep,
) -> ProjectSchema:
    """Retrieve the configuration of the currently active project."""
    return project_service.get_active_project_info()


@projects_router.get(
    path="",
    tags=["Projects"],
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
def get_projects_list(
    project_service: ProjectServiceDep,
    offset: Annotated[int, Query(ge=0, le=1000)] = 0,
    limit: Annotated[int, Query(ge=0, le=1000)] = 20,
) -> ProjectsListSchema:
    """Retrieve a list of all available project configurations."""
    return project_service.list_projects(offset=offset, limit=limit)


@projects_router.get(
    path="/{project_id}",
    tags=["Projects"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved the configuration for a project.",
        },
        status.HTTP_404_NOT_FOUND: {"description": "Project not found."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while retrieving the configuration of a project.",
        },
    },
)
def get_project(
    project_id: UUID,
    project_service: ProjectServiceDep,
) -> ProjectSchema:
    """Retrieve the project's configuration."""
    return project_service.get_project(project_id)


@projects_router.put(
    path="/{project_id}",
    tags=["Projects"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully updated the configuration for the project.",
        },
        status.HTTP_404_NOT_FOUND: {"description": "Project not found."},
        status.HTTP_409_CONFLICT: {"description": "Project name already exists."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while updating the configuration of the project.",
        },
    },
)
def update_project(
    project_id: UUID,
    payload: ProjectUpdateSchema,
    project_service: ProjectServiceDep,
) -> ProjectSchema:
    """Update the project's configuration."""
    return project_service.update_project(project_id=project_id, update_data=payload)


@projects_router.get(
    path="/export",
    tags=["Projects"],
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
    if names:
        logger.debug(f"Exporting projects with names: {names}")

    # Placeholder for future service integration.

    return Response(status_code=status.HTTP_200_OK, media_type="application/zip", content=b"")


@projects_router.post(
    path="/import",
    tags=["Projects"],
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
    # Placeholder for future service integration.

    return Response(status_code=status.HTTP_201_CREATED)
