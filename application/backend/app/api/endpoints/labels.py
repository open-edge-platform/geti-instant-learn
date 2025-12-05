# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated
from uuid import UUID

from fastapi import Query, Response, status

from api.routers import projects_router
from dependencies import LabelServiceDep
from domain.services.schemas.label import LabelCreateSchema, LabelSchema, LabelsListSchema, LabelUpdateSchema

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
    label_service: LabelServiceDep,
) -> Response:
    """Create a new label with the given name."""
    label = label_service.create_label(project_id=project_id, create_data=payload)
    logger.info("Successfully created '%s' label with id %s", label.name, label.id)

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
        status.HTTP_404_NOT_FOUND: {"description": "Project or label not found."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Unexpected error occurred."},
    },
)
def get_label_by_id(
    project_id: UUID,
    label_id: UUID,
    label_service: LabelServiceDep,
) -> LabelSchema:
    """Get a label by its ID for selected project."""
    return label_service.get_label_by_id(project_id=project_id, label_id=label_id)


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
    label_service: LabelServiceDep,
    offset: Annotated[int, Query(ge=0, le=1000)] = 0,
    limit: Annotated[int, Query(ge=0, le=1000)] = 20,
) -> LabelsListSchema:
    """Get all labels for selected project"""
    return label_service.get_all_labels(project_id=project_id, offset=offset, limit=limit)


@projects_router.delete(
    path="/{project_id}/labels/{label_id}",
    tags=["Labels"],
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Successfully deleted the label.",
        },
        status.HTTP_403_FORBIDDEN: {"description": "Label is being used in prompts and cannot be deleted."},
        status.HTTP_404_NOT_FOUND: {"description": "Project or label not found."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while deleting the label.",
        },
    },
)
def delete_label_by_id(
    project_id: UUID,
    label_id: UUID,
    label_service: LabelServiceDep,
) -> Response:
    """Delete a label by its ID for selected project."""
    label_service.delete_label(project_id=project_id, label_id=label_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@projects_router.put(
    path="/{project_id}/labels/{label_id}",
    tags=["Labels"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully updated the label.",
        },
        status.HTTP_404_NOT_FOUND: {"description": "Project or label not found."},
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
    label_service: LabelServiceDep,
) -> LabelSchema:
    """
    Update the label.
    """
    return label_service.update_label(project_id=project_id, label_id=label_id, update_data=payload)
