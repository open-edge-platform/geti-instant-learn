# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from fastapi import status

from api.routers import projects_router
from dependencies import PipelineManagerDep, ProjectServiceDep
from domain.services.schemas.model_status import ModelStatusSchema


@projects_router.get(
    path="/{project_id}/model-status",
    tags=["Model Status"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Current model busy-state for the project."},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found."},
    },
)
def get_model_status(
    project_id: UUID, project_service: ProjectServiceDep, pipeline_manager: PipelineManagerDep
) -> ModelStatusSchema:
    """Return the current processor load state for the given project."""
    project_service.get_project(project_id)  # validates project_id, 404 if missing
    return pipeline_manager.get_model_status()
