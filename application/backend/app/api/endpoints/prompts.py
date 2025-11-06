# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import Response, status

from api.routers import projects_router
from dependencies import PromptServiceDep
from domain.services.schemas.prompt import PromptCreateSchema, PromptSchema, PromptsListSchema, PromptUpdateSchema

logger = logging.getLogger(__name__)


@projects_router.get(
    path="/{project_id}/prompts",
    tags=["Prompts"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successfully retrieved the list of all prompts for the project."},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Unexpected error occurred."},
    },
)
def get_all_prompts(
    project_id: UUID, prompt_service: PromptServiceDep, offset: int = 0, limit: int = 10
) -> PromptsListSchema:
    """
    Retrieve a list of all prompts for the project with pagination.
    """
    limit = min(limit, 100)  # set the maximum limit to prevent excessive queries
    return prompt_service.list_prompts(project_id, offset=offset, limit=limit)


@projects_router.get(
    path="/{project_id}/prompts/{prompt_id}",
    tags=["Prompts"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successfully retrieved the details of the prompt."},
        status.HTTP_404_NOT_FOUND: {"description": "Project or prompt not found."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Unexpected error occurred."},
    },
)
def get_prompt(project_id: UUID, prompt_id: UUID, prompt_service: PromptServiceDep) -> PromptSchema:
    """
    Retrieve the details of a specific prompt.
    """
    return prompt_service.get_prompt(project_id, prompt_id)


@projects_router.post(
    path="/{project_id}/prompts",
    tags=["Prompts"],
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Successfully created a new prompt."},
        status.HTTP_404_NOT_FOUND: {"description": "Project, label, or frame not found."},
        status.HTTP_409_CONFLICT: {"description": "Prompt already exists."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Unexpected error occurred."},
    },
)
def create_prompt(project_id: UUID, payload: PromptCreateSchema, prompt_service: PromptServiceDep) -> PromptSchema:
    """
    Create a new text or visual prompt for the project.
    Text prompts are limited to one per project.
    Visual prompts must reference an existing frame and include annotations.
    Returns the created prompt.
    """
    return prompt_service.create_prompt(project_id=project_id, create_data=payload)


@projects_router.put(
    path="/{project_id}/prompts/{prompt_id}",
    tags=["Prompts"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successfully updated the prompt."},
        status.HTTP_404_NOT_FOUND: {"description": "Project, prompt, label, or frame not found."},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid update data or type mismatch."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Unexpected error occurred."},
    },
)
def update_prompt(
    project_id: UUID, prompt_id: UUID, payload: PromptUpdateSchema, prompt_service: PromptServiceDep
) -> PromptSchema:
    """
    Update an existing prompt (text or visual) for the project.
    """
    return prompt_service.update_prompt(project_id=project_id, prompt_id=prompt_id, update_data=payload)


@projects_router.delete(
    path="/{project_id}/prompts/{prompt_id}",
    tags=["Prompts"],
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Successfully deleted the prompt."},
        status.HTTP_404_NOT_FOUND: {"description": "Project or prompt not found."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Unexpected error occurred."},
    },
)
def delete_prompt(project_id: UUID, prompt_id: UUID, prompt_service: PromptServiceDep) -> Response:
    """
    Delete a prompt from the project.
    """
    prompt_service.delete_prompt(project_id=project_id, prompt_id=prompt_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
