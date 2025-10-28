# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Annotated
from uuid import UUID

from fastapi import HTTPException, Query, Response, status

from dependencies import ConfigChangeDispatcherDep, SessionDep
from routers import projects_router
from services.errors import ResourceAlreadyExistsError, ResourceNotFoundError
from services.label import LabelService
from services.schemas.label import LabelCreateSchema, LabelSchema, LabelsListSchema, LabelUpdateSchema

logger = logging.getLogger(__name__)


@projects_router.post(
    path="/{project_id}/labels",
    tags=["Labels"],
    responses={
        status.HTTP_201_CREATED: {"description": "Successfully created a new label."},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found."},
        status.HTTP_409_CONFLICT: {"description": "Label with this name already exists."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Unexpected error occurred."},
    },
)
def create_label(
    project_id: UUID,
    payload: LabelCreateSchema,
    db_session: SessionDep,
    config_dispatcher: ConfigChangeDispatcherDep,
) -> Response:
    """Create a new label with the given name."""

    logger.debug(f"Attempting to create label with name: {payload.name}")
    service = LabelService(session=db_session, config_change_dispatcher=config_dispatcher)
    try:
        label = service.create_label(project_id=project_id, create_data=payload)
        logger.info(f"Successfully created '{label.name}' label with id {label.id}")
    except ResourceAlreadyExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during label creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create a label due to internal server error.",
        )

    return Response(
        status_code=status.HTTP_201_CREATED,
        headers={"Location": f"/projects/{project_id}/labels/{label.id}"},
        content=label.model_dump_json(),
        media_type="application/json",
    )


@projects_router.get(
    path="/{project_id}/labels/{label_id}",
    tags=["Labels"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successfully retrieved the details of label."},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Unexpected error occurred."},
    },
)
def get_label_by_id(
    project_id: UUID,
    label_id: UUID,
    db_session: SessionDep,
    config_dispatcher: ConfigChangeDispatcherDep,
) -> LabelSchema:
    """Get a label by its ID for selected project."""
    service = LabelService(session=db_session, config_change_dispatcher=config_dispatcher)
    try:
        return service.get_label_by_id(project_id=project_id, label_id=label_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception:
        logger.exception(f"Internal error retrieving label id={label_id} for project id {project_id}")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve label due to internal server error.",
        )


@projects_router.get(
    path="/{project_id}/labels",
    tags=["Labels"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved a list of all labels for selected project.",
        },
        status.HTTP_404_NOT_FOUND: {"description": "Project not found."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while retrieving available project configurations.",
        },
    },
)
def get_all_labels(
    project_id: UUID,
    db_session: SessionDep,
    config_dispatcher: ConfigChangeDispatcherDep,
    offset: Annotated[int, Query(ge=0, le=1000)] = 0,
    limit: Annotated[int, Query(ge=0, le=1000)] = 20,
) -> LabelsListSchema:
    """Get all labels for selected project"""
    service = LabelService(session=db_session, config_change_dispatcher=config_dispatcher)
    try:
        return service.get_all_labels(project_id=project_id, offset=offset, limit=limit)
    except Exception:
        logger.exception(f"Internal error listing labels for project id {project_id}")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list labels due to internal server error.",
        )


@projects_router.delete(
    path="/{project_id}/labels/{label_id}",
    tags=["Labels"],
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Successfully deleted the label.",
        },
        status.HTTP_404_NOT_FOUND: {"description": "Project not found."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while deleting the label.",
        },
    },
)
def delete_label_by_id(
    project_id: UUID,
    label_id: UUID,
    db_session: SessionDep,
    config_dispatcher: ConfigChangeDispatcherDep,
) -> Response:
    """Delete a label by its ID for selected project."""
    service = LabelService(session=db_session, config_change_dispatcher=config_dispatcher)
    try:
        service.delete_label(project_id=project_id, label_id=label_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception:
        logger.exception(f"Error deleting label {label_id} for project {project_id}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete label.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@projects_router.put(
    path="/{project_id}/labels/{label_id}",
    tags=["Labels"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully updated the label.",
        },
        status.HTTP_404_NOT_FOUND: {"description": "Project not found."},
        status.HTTP_409_CONFLICT: {"description": "Label name already exists."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while updating the label.",
        },
    },
)
def update_label(
    project_id: UUID,
    label_id: UUID,
    payload: LabelUpdateSchema,
    db_session: SessionDep,
    config_dispatcher: ConfigChangeDispatcherDep,
) -> LabelSchema:
    """
    Update the label.
    """
    logger.debug(f"Received PUT label {label_id} for {project_id} request.")
    service = LabelService(session=db_session, config_change_dispatcher=config_dispatcher)
    try:
        return service.update_label(project_id=project_id, label_id=label_id, update_data=payload)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ResourceAlreadyExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception:
        logger.exception(f"Internal error updating label with id={label_id} for project id {project_id}")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update label due to internal server error.",
        )
