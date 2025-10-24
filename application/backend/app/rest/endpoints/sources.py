# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import HTTPException, Response, status

from dependencies import ConfigChangeDispatcherDep, SessionDep
from routers import projects_router
from services.errors import ResourceNotFoundError, ResourceUpdateConflictError
from services.schemas.source import SourceCreateSchema, SourceSchema, SourcesListSchema, SourceUpdateSchema
from services.source import SourceService

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
def get_sources(
    project_id: UUID, db_session: SessionDep, config_dispatcher: ConfigChangeDispatcherDep
) -> SourcesListSchema:
    """
    Retrieve the source configuration of the project.
    """
    logger.debug(f"Received GET project {project_id} sources request.")
    service = SourceService(session=db_session, config_change_dispatcher=config_dispatcher)
    try:
        return service.list_sources(project_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception(f"Error listing sources for project {project_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list sources.")


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
    db_session: SessionDep,
    config_dispatcher: ConfigChangeDispatcherDep,
) -> SourceSchema:
    """
    Create a new source configuration for the project.
    """
    logger.debug(f"Received POST source request for project {project_id} with payload: {payload}.")
    service = SourceService(session=db_session, config_change_dispatcher=config_dispatcher)
    try:
        return service.create_source(project_id=project_id, create_data=payload)
    except ResourceUpdateConflictError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception:
        logger.exception(f"Error creating source for project {project_id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create source.")


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
    db_session: SessionDep,
    config_dispatcher: ConfigChangeDispatcherDep,
) -> SourceSchema:
    """
    Update the project's source configuration.
    """
    logger.debug(f"Received PUT source {source_id} request for project {project_id}.")
    service = SourceService(session=db_session, config_change_dispatcher=config_dispatcher)
    try:
        return service.update_source(project_id=project_id, source_id=source_id, update_data=payload)
    except ResourceUpdateConflictError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception(f"Error upserting source {source_id} for project {project_id}: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update source configuration.")


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
    db_session: SessionDep,
    config_dispatcher: ConfigChangeDispatcherDep,
) -> Response:
    """
    Delete the specified project's source configuration.
    """
    logger.debug(f"Received DELETE source {source_id} request for project {project_id}.")
    service = SourceService(session=db_session, config_change_dispatcher=config_dispatcher)
    try:
        service.delete_source(project_id=project_id, source_id=source_id)
    except ResourceNotFoundError:
        logger.warning(f"Source with id {source_id} not found during delete operation.")
    except Exception as e:
        logger.exception(f"Error deleting source {source_id} for project {project_id}: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete source.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
