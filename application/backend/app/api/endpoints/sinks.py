# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import Response, status

from api.routers import projects_router
from dependencies import SinkServiceDep
from domain.services.schemas.sink import SinkCreateSchema, SinkSchema, SinksListSchema, SinkUpdateSchema

logger = logging.getLogger(__name__)


@projects_router.get(
    path="/{project_id}/sinks",
    tags=["Sinks"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved the sink configuration for the project.",
            "content": {
                "application/json": {
                    "example": {
                        "sinks": [
                            {
                                "id": "550e8400-e29b-41d4-a716-446655440000",
                                "name": "Active Sink",
                                "connected": True,
                                "config": {
                                    "sink_type": "mqtt",
                                    "broker_host": "localhost",
                                    "broker_port": 1883,
                                    "topic": "predictions",
                                    "auth_required": True,
                                    "output_formats": ["predictions"],
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
def get_sinks(project_id: UUID, sink_service: SinkServiceDep) -> SinksListSchema:
    """
    Retrieve the sink configuration of the project.
    """
    return sink_service.list_sinks(project_id=project_id)


@projects_router.put(
    path="/{project_id}/sinks/{sink_id}",
    tags=["Sinks"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Sink configuration updated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "name": "Updated Sink",
                        "connected": True,
                        "config": {
                            "sink_type": "mqtt",
                            "broker_host": "localhost",
                            "broker_port": 1883,
                            "topic": "predictions",
                            "auth_required": True,
                            "output_formats": ["predictions"],
                        },
                    },
                }
            },
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Project or sink configuration not found",
            "content": {
                "application/json": {
                    "examples": {
                        "project_missing": {
                            "summary": "Project not found",
                            "value": {"detail": "Project with ID 3fa85f64-5717-4562-b3fc-2c963f66afa6 not found."},
                        },
                        "sink_missing": {
                            "summary": "Sink configuration not found",
                            "value": {"detail": "Sink with ID 04b34cb0-c405-4566-990a-4eaeeaaa515a not found."},
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
def update_sink(project_id: UUID, sink_id: UUID, payload: SinkUpdateSchema, sink_service: SinkServiceDep) -> SinkSchema:
    """
    Update the project's sink configuration.
    """
    return sink_service.update_sink(project_id=project_id, sink_id=sink_id, update_data=payload)


@projects_router.delete(
    path="/{project_id}/sinks/{sink_id}",
    tags=["Sinks"],
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Sink configuration deleted successfully"},
        status.HTTP_404_NOT_FOUND: {
            "description": "Project or sink configuration not found",
            "content": {
                "application/json": {
                    "examples": {
                        "project_missing": {
                            "summary": "Project not found",
                            "value": {"detail": "Project with ID 3fa85f64-5717-4562-b3fc-2c963f66afa6 not found."},
                        },
                        "sink_missing": {
                            "summary": "Sink configuration not found",
                            "value": {"detail": "Sink with ID 04b34cb0-c405-4566-990a-4eaeeaaa515a not found."},
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
def delete_sink(project_id: UUID, sink_id: UUID, sink_service: SinkServiceDep) -> Response:
    """
    Delete the specified project's sink configuration.
    """
    sink_service.delete_sink(project_id=project_id, sink_id=sink_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@projects_router.post(
    path="/{project_id}/sinks",
    tags=["Sinks"],
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {
            "description": "Sink configuration created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "name": "New Sink",
                        "connected": False,
                        "config": {
                            "sink_type": "mqtt",
                            "broker_host": "localhost",
                            "broker_port": 1883,
                            "topic": "predictions",
                            "auth_required": True,
                            "output_formats": ["predictions"],
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
            "description": "Invalid sink configuration data",
            "content": {"application/json": {"example": {"detail": "Invalid sink configuration data was provided."}}},
        },
        status.HTTP_409_CONFLICT: {
            "description": "Conflicting data was provided",
            "content": {
                "application/json": {
                    "example": {"detail": "A sink configuration with the name 'Sink' already exists in this project."}
                }
            },
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred",
            "content": {
                "application/json": {
                    "example": {"detail": "An unexpected error occurred while creating the sink configuration"}
                }
            },
        },
    },
)
def create_sink(project_id: UUID, payload: SinkCreateSchema, source_service: SinkServiceDep) -> SinkSchema:
    """
    Create a new sink configuration for the project.
    """
    return source_service.create_sink(project_id=project_id, create_data=payload)
