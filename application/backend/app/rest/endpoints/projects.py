# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated
from uuid import UUID

from fastapi import HTTPException, Query, Response, status

from dependencies import SessionDep
from rest.mappers.project import (
    project_db_to_schema,
    project_post_payload_to_db,
    projects_db_to_list_items,
)
from routers import projects_router
from services.common import ResourceAlreadyExistsError, ResourceNotFoundError
from services.project import ProjectService
from services.schemas.project import (
    ProjectPostPayload,
    ProjectPutPayload,
    ProjectSchema,
    ProjectsListSchema,
)

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
    service = ProjectService(db_session)
    project_entity = project_post_payload_to_db(payload)
    try:
        project = service.create_project(project_entity)
        logger.info(f"Successfully created '{project.name}' project with id {project.id}")
    except ResourceAlreadyExistsError as e:
        logger.error(f"Project creation failed: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during project creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create a project due to internal server error.",
        )

    return Response(status_code=status.HTTP_201_CREATED, headers={"Location": f"/projects/{project.id}"})


@projects_router.delete(
    path="/{project_id}",
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
def delete_project(project_id: UUID, db_session: SessionDep) -> Response:
    """
    Delete the specified project.
    """
    logger.debug(f"Received DELETE project {project_id} request.")
    service = ProjectService(db_session)
    try:
        service.delete_project(project_id)
    except ResourceNotFoundError:
        logger.warning(f"Project with id {project_id} not found during delete operation.")
    except Exception as e:
        logger.exception(f"Error occurred while deleting project with id {project_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete project with id {project_id} due to an internal error.",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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
    service = ProjectService(db_session)
    try:
        project = service.get_active_project()
        return project_db_to_schema(project)
    except ResourceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active project found.",
        )
    except Exception as e:
        logger.exception(f"Internal error fetching active project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve active project.",
        )


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
    service = ProjectService(db_session)
    try:
        projects = service.list_projects()
        items = projects_db_to_list_items(projects)
        logger.debug(f"Returning {len(items)} projects")
        return ProjectsListSchema(projects=items)
    except Exception as e:
        logger.exception(f"Internal error listing projects: {e}")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list projects.",
        )


@projects_router.get(
    path="/{project_id}",
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
def get_project(project_id: UUID, db_session: SessionDep) -> ProjectSchema:
    """
    Retrieve the project's configuration.
    """
    logger.debug(f"Received GET project {project_id} request.")
    service = ProjectService(db_session)
    try:
        project = service.get_project(project_id)
        return project_db_to_schema(project)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception(f"Internal error retrieving project id={project_id}: {e}")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve project.",
        )


@projects_router.put(
    path="/{project_id}",
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
def update_project(project_id: UUID, payload: ProjectPutPayload, db_session: SessionDep) -> ProjectSchema:
    """
    Update the project's configuration.
    """
    logger.debug(f"Received PUT project {project_id} request.")
    service = ProjectService(db_session)
    try:
        project = service.update_project(project_id=project_id, new_name=payload.name)
        return project_db_to_schema(project)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ResourceAlreadyExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        logger.exception(f"Internal error updating project with id={project_id}: {e}")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update project due to internal server error.",
        )


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

    # Placeholder for future service integration.

    return Response(status_code=status.HTTP_200_OK, media_type="application/zip", content=b"")


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

    # Placeholder for future service integration.

    return Response(status_code=status.HTTP_201_CREATED)
