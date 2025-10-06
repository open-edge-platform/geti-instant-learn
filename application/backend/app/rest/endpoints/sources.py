# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import HTTPException, Response, status

from dependencies import SessionDep
from rest.schemas.source import SourcePayloadSchema, SourceSchema, SourcesListSchema
from routers import projects_router
from services.common import ResourceNotFoundError, ResourceUpdateConflictError
from services.source import SourceService

logger = logging.getLogger(__name__)


@projects_router.get(
    path="/{project_id}/sources",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successfully retrieved the sources configuration for the project."},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Unexpected error occurred."},
    },
)
def get_sources(project_id: UUID, db_session: SessionDep) -> SourcesListSchema:
    """
    Retrieve the source configuration of the project.
    """
    logger.debug(f"Received GET project {project_id} sources request.")
    service = SourceService(db_session)
    try:
        return service.list_sources(project_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception(f"Error listing sources for project {project_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list sources.")


@projects_router.put(
    path="/{project_id}/sources/{source_id}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_201_CREATED: {"description": "Successfully created the configuration for the project's source."},
        status.HTTP_200_OK: {"description": "Successfully updated the configuration for the project's source."},
        status.HTTP_404_NOT_FOUND: {"description": "Project or source not found."},
        status.HTTP_409_CONFLICT: {"description": "Source type change is not allowed."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Unexpected error occurred."},
    },
)
def update_source(
    project_id: UUID, source_id: UUID, payload: SourcePayloadSchema, db_session: SessionDep, response: Response
) -> SourceSchema:
    """
    Update the project's source configuration.
    """
    logger.debug(f"Received PUT source {source_id} request for project {project_id}.")
    service = SourceService(db_session)
    try:
        source, created = service.upsert_source(project_id=project_id, source_id=source_id, payload=payload)
        response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
        return source
    except ResourceUpdateConflictError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception(f"Error upserting source {source_id} for project {project_id}: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update source configuration.")


@projects_router.delete(
    path="/{project_id}/sources/{source_id}",
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
def delete_source(project_id: UUID, source_id: UUID, db_session: SessionDep) -> Response:
    """
    Delete the specified project's source configuration.
    """
    logger.debug(f"Received DELETE source {source_id} request for project {project_id}.")
    service = SourceService(db_session)
    try:
        service.delete_source(project_id=project_id, source_id=source_id)
    except ResourceNotFoundError:
        logger.warning(f"Source with id {source_id} not found during delete operation.")
    except Exception as e:
        logger.exception(f"Error deleting source {source_id} for project {project_id}: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete source.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
