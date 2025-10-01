# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated
from uuid import UUID

from fastapi import Form, Response, status

from routers import projects_router

logger = logging.getLogger(__name__)


@projects_router.post(
    path="/{project_id}/prompts",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {
            "description": "Successfully added a new prompt to the project.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while adding a new prompt to the project.",
        },
    },
)
def create_prompt(
    project_id: UUID,
    prompt_name: Annotated[str, Form()],
    prompt_type: Annotated[str, Form()],
    # image: Annotated[UploadFile, File()],
    # annotation_data: Annotated[UploadFile, File()]
) -> Response:
    """
    Add a new prompt to the project.
    """
    logger.debug(f"Received POST project {project_id} prompt request, name: {prompt_name}, type: {prompt_type}")

    return Response(status_code=status.HTTP_201_CREATED)


@projects_router.delete(
    path="/{project_id}/prompts/{prompt_id}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully deleted the prompt's directory and all its contents.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while deleting the prompt's directory.",
        },
    },
)
def delete_prompt(project_id: UUID, prompt_id: UUID) -> Response:
    """
    Delete the prompt's directory and all its contents.
    """
    logger.debug(f"Received DELETE project {project_id} prompt {prompt_id} request.")

    return Response(status_code=status.HTTP_200_OK)


@projects_router.get(
    path="/{project_id}/prompts/{prompt_id}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved the details and files of the prompt.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while retrieving the prompt details.",
        },
    },
)
def get_prompt(project_id: UUID, prompt_id: UUID) -> Response:
    """
    Retrieve the details and files of the prompt.
    """
    logger.debug(f"Received GET project {project_id} prompt {prompt_id} request.")

    return Response(status_code=status.HTTP_200_OK)


@projects_router.get(
    path="/{project_id}/prompts",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved the list of all prompts for the project.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while retrieving the list of prompts for the project.",
        },
    },
)
def get_all_prompts(project_id: UUID) -> Response:
    """
    Retrieve a list of all prompts for the project.
    """
    logger.debug(f"Received GET project {project_id} prompts request.")

    return Response(status_code=status.HTTP_200_OK, content={"project_prompts": []})


@projects_router.put(
    path="/{project_id}/prompts/{prompt_id}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully updated the files for the prompt.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while updating the prompt details.",
        },
    },
)
def update_prompt(project_id: UUID, prompt_id: UUID) -> Response:
    """
    Update the existing files of the prompt.
    """
    logger.debug(f"Received PUT project {project_id} prompt {prompt_id} request.")

    return Response(status_code=status.HTTP_200_OK)
