# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated
from uuid import UUID

from fastapi import Query, Response, status

from api.routers import projects_router
from dependencies import ModelServiceDep
from domain.services.schemas.processor import (
    ProcessorCreateSchema,
    ProcessorListSchema,
    ProcessorSchema,
    ProcessorUpdateSchema,
)

logger = logging.getLogger(__name__)


@projects_router.get(
    path="/{project_id}/models",
    tags=["Models"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved the active model configuration for the project.",
            "content": {
                "application/json": {
                    "example": {
                        "models": [
                            {
                                "id": "550e8400-e29b-41d4-a716-446655440000",
                                "name": "Active Model",
                                "active": True,
                                "config": {
                                    "confidence_threshold": 0.38,
                                    "model_type": "matcher",
                                    "num_background_points": 3,
                                    "num_foreground_points": 5,
                                    "precision": "bf16",
                                    "sam_model": "SAM-HQ-tiny",
                                    "encoder_model": "dinov3_small",
                                    "use_mask_refinement": False,
                                },
                            }
                        ]
                    },
                }
            },
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Project not found",
            "content": {
                "application/json": {
                    "examples": {
                        "project_missing": {
                            "summary": "Project not found",
                            "value": {"detail": "Project with ID 123e4567-e89b-12d3-a456-426614174000 not found."},
                        }
                    }
                }
            },
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "An unexpected error occurred while retrieving the active model configuration"
                    }
                }
            },
        },
    },
)
def get_all_models(
    project_id: UUID,
    model_service: ModelServiceDep,
    offset: Annotated[int, Query(ge=0, le=1000)] = 0,
    limit: Annotated[int, Query(ge=0, le=1000)] = 20,
) -> ProcessorListSchema:
    """
    Retrieve the all model configurations of the project.
    """
    return model_service.list_models(project_id=project_id, offset=offset, limit=limit)


# needs to be before /{project_id}/models/{model_id} to avoid path conflict
@projects_router.get(
    path="/{project_id}/models/active",
    tags=["Models"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved the active model configuration for the project.",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "name": "Active Model",
                        "active": True,
                        "config": {
                            "confidence_threshold": 0.38,
                            "model_type": "matcher",
                            "num_background_points": 3,
                            "num_foreground_points": 5,
                            "precision": "bf16",
                            "sam_model": "SAM-HQ-tiny",
                            "encoder_model": "dinov3_small",
                            "use_mask_refinement": False,
                        },
                    },
                }
            },
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Project or active model configuration not found",
            "content": {
                "application/json": {
                    "examples": {
                        "project_missing": {
                            "summary": "Project not found",
                            "value": {"detail": "Project with ID 123e4567-e89b-12d3-a456-426614174000 not found."},
                        },
                        "no_active_model": {
                            "summary": "No active model",
                            "value": {"detail": "No active model configuration found for specified project"},
                        },
                    }
                }
            },
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "An unexpected error occurred while retrieving the active model configuration"
                    }
                }
            },
        },
    },
)
def get_active_model(project_id: UUID, model_service: ModelServiceDep) -> ProcessorSchema:
    """
    Retrieve the active model configuration of the project.
    """
    return model_service.get_active_model(project_id)


@projects_router.get(
    path="/{project_id}/models/{model_id}",
    tags=["Models"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved the model configuration for the project.",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "name": "My Model",
                        "active": False,
                        "config": {
                            "confidence_threshold": 0.38,
                            "model_type": "matcher",
                            "num_background_points": 3,
                            "num_foreground_points": 5,
                            "precision": "bf16",
                            "sam_model": "SAM-HQ-tiny",
                            "encoder_model": "dinov3_small",
                            "use_mask_refinement": False,
                        },
                    },
                }
            },
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Project or model configuration not found",
            "content": {
                "application/json": {
                    "examples": {
                        "project_missing": {
                            "summary": "Project not found",
                            "value": {"detail": "Project with id 123e4567-e89b-12d3-a456-426614174000 not found."},
                        },
                        "model_missing": {
                            "summary": "Model configuration not found",
                            "value": {"detail": "No active model configuration found for the specified project."},
                        },
                    }
                }
            },
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred",
            "content": {
                "application/json": {
                    "example": {"detail": "An unexpected error occurred while retrieving the model configuration"}
                }
            },
        },
    },
)
def get_model(project_id: UUID, model_id: UUID, model_service: ModelServiceDep) -> ProcessorSchema:
    """
    Retrieve the model configuration of the project.
    """
    return model_service.get_model(project_id=project_id, model_id=model_id)


@projects_router.post(
    path="/{project_id}/models",
    tags=["Models"],
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {
            "description": "Model configuration created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "name": "New Model",
                        "active": False,
                        "config": {
                            "confidence_threshold": 0.38,
                            "model_type": "matcher",
                            "num_background_points": 3,
                            "num_foreground_points": 5,
                            "precision": "bf16",
                            "sam_model": "SAM-HQ-tiny",
                            "encoder_model": "dinov3_small",
                            "use_mask_refinement": False,
                        },
                    },
                }
            },
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Project not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Project with id 123e4567-e89b-12d3-a456-426614174000 not found."}
                }
            },
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "Invalid model configuration data",
            "content": {"application/json": {"example": {"detail": "Invalid model configuration data was provided."}}},
        },
        status.HTTP_409_CONFLICT: {
            "description": "Conflicting data was provided",
            "content": {
                "application/json": {
                    "example": {"detail": "A model configuration with the name 'Model' already exists in this project."}
                }
            },
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred",
            "content": {
                "application/json": {
                    "example": {"detail": "An unexpected error occurred while creating the model configuration"}
                }
            },
        },
    },
)
def create_model(project_id: UUID, payload: ProcessorCreateSchema, model_service: ModelServiceDep) -> Response:
    """
    Create a new model configuration for the project.
    """
    model = model_service.create_model(project_id=project_id, create_data=payload)
    return Response(
        status_code=status.HTTP_201_CREATED,
        headers={"Location": f"/projects/{project_id}/models/{model.id}"},
        content=model.model_dump_json(),
        media_type="application/json",
    )


@projects_router.put(
    path="/{project_id}/models/{model_id}",
    tags=["Models"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Model configuration updated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "name": "Updated Model",
                        "active": True,
                        "config": {
                            "confidence_threshold": 0.45,
                            "model_type": "matcher",
                            "num_background_points": 3,
                            "num_foreground_points": 5,
                            "precision": "fp16",
                            "sam_model": "SAM-HQ-tiny",
                            "encoder_model": "dinov3_small",
                            "use_mask_refinement": False,
                        },
                    },
                }
            },
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Project or model configuration not found",
            "content": {
                "application/json": {
                    "examples": {
                        "project_missing": {
                            "summary": "Project not found",
                            "value": {"detail": "Project with ID 123e4567-e89b-12d3-a456-426614174000 not found."},
                        },
                        "model_missing": {
                            "summary": "Model configuration not found",
                            "value": {"detail": "No active model configuration found for the specified project."},
                        },
                    }
                }
            },
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "Invalid update data",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Invalid model configuration update: "
                        "precision must be one of ['bf16', 'fp16', 'fp32']"
                    }
                }
            },
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred",
            "content": {
                "application/json": {
                    "example": {"detail": "An unexpected error occurred while updating the model configuration"}
                }
            },
        },
    },
)
def update_model(
    project_id: UUID, model_id: UUID, payload: ProcessorUpdateSchema, model_service: ModelServiceDep
) -> ProcessorSchema:
    """
    Update an existing model configuration for the project.
    """
    return model_service.update_model(project_id=project_id, model_id=model_id, update_data=payload)


@projects_router.delete(
    path="/{project_id}/models/{model_id}",
    tags=["Models"],
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Model configuration deleted successfully"},
        status.HTTP_404_NOT_FOUND: {
            "description": "Project or model configuration not found",
            "content": {
                "application/json": {
                    "examples": {
                        "project_missing": {
                            "summary": "Project not found",
                            "value": {"detail": "Project with ID 3fa85f64-5717-4562-b3fc-2c963f66afa6 not found."},
                        },
                        "model_missing": {
                            "summary": "Model configuration not found",
                            "value": {"detail": "Processor with ID 04b34cb0-c405-4566-990a-4eaeeaaa515a not found."},
                        },
                    }
                }
            },
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred",
            "content": {
                "application/json": {
                    "example": {"detail": "An unexpected error occurred while deleting the model configuration"}
                }
            },
        },
    },
)
def delete_model(project_id: UUID, model_id: UUID, model_service: ModelServiceDep) -> Response:
    """
    Delete a model configuration from the project.
    """
    model_service.delete_model(project_id=project_id, model_id=model_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
