# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from domain.db.models import PromptType
from domain.services.schemas.annotation import AnnotationSchema
from domain.services.schemas.base import BaseIDPayload, BaseIDSchema, Pagination


class TextPromptCreateSchema(BaseIDPayload):
    """Schema for creating a text prompt."""

    type: Literal[PromptType.TEXT]  # type: ignore[valid-type]
    content: str = Field(..., description="Text content of the prompt", min_length=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "TEXT",
                "content": "red car",
            }
        }
    }


class VisualPromptCreateSchema(BaseIDPayload):
    """Schema for creating a visual prompt."""

    type: Literal[PromptType.VISUAL]  # type: ignore[valid-type]
    frame_id: UUID = Field(..., description="ID of the frame to use for the prompt")
    annotations: list[AnnotationSchema] = Field(..., description="List of annotations for the prompt", min_length=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "VISUAL",
                "frame_id": "123e4567-e89b-12d3-a456-426614174000",
                "annotations": [
                    {
                        "config": {
                            "type": "polygon",
                            "points": [
                                {"x": 0.1, "y": 0.1},
                                {"x": 0.5, "y": 0.1},
                                {"x": 0.5, "y": 0.5},
                                {"x": 0.1, "y": 0.5},
                            ],
                        },
                        "label_id": "123e4567-e89b-12d3-a456-426614174001",
                    }
                ],
            }
        }
    }


PromptCreateSchema = Annotated[VisualPromptCreateSchema | TextPromptCreateSchema, Field(discriminator="type")]


class TextPromptUpdateSchema(BaseModel):
    """Schema for updating a text prompt."""

    type: Literal[PromptType.TEXT]  # type: ignore[valid-type]
    content: str | None = Field(None, description="Text content of the prompt", min_length=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "TEXT",
                "content": "red car",
            }
        }
    }


class VisualPromptUpdateSchema(BaseModel):
    """Schema for updating a visual prompt."""

    type: Literal[PromptType.VISUAL]  # type: ignore[valid-type]
    frame_id: UUID | None = Field(None, description="ID of the frame to use for the prompt")
    annotations: list[AnnotationSchema] | None = Field(
        None, description="List of annotations for the prompt", min_length=1
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "VISUAL",
                "frame_id": "123e4567-e89b-12d3-a456-426614174000",
                "annotations": [
                    {
                        "config": {
                            "type": "polygon",
                            "points": [
                                {"x": 0.1, "y": 0.1},
                                {"x": 0.5, "y": 0.1},
                                {"x": 0.5, "y": 0.5},
                                {"x": 0.1, "y": 0.5},
                            ],
                        },
                        "label_id": "123e4567-e89b-12d3-a456-426614174001",
                    }
                ],
            }
        }
    }


PromptUpdateSchema = Annotated[VisualPromptUpdateSchema | TextPromptUpdateSchema, Field(discriminator="type")]


class TextPromptSchema(BaseIDSchema):
    """Schema for a text prompt response."""

    type: Literal[PromptType.TEXT]  # type: ignore[valid-type]
    content: str


class VisualPromptSchema(BaseIDSchema):
    """Schema for a visual prompt response."""

    type: Literal[PromptType.VISUAL]  # type: ignore[valid-type]
    frame_id: UUID
    annotations: list[AnnotationSchema]


PromptSchema = Annotated[VisualPromptSchema | TextPromptSchema, Field(discriminator="type")]


class PromptsListSchema(BaseModel):
    """Schema for listing prompts."""

    prompts: list[PromptSchema]
    pagination: Pagination = Field(default_factory=lambda: Pagination(count=0, total=0, offset=0, limit=20))
