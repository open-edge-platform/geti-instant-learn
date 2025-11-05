# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from services.schemas.annotation import Annotation
from services.schemas.base import BaseIDPayload, BaseIDSchema, Pagination


class PromptType(StrEnum):
    """Enum for different types of prompts."""

    TEXT = "TEXT"
    VISUAL = "VISUAL"


class TextPromptCreateSchema(BaseIDPayload):
    """Schema for creating a text prompt."""

    type: Literal[PromptType.TEXT]
    content: str = Field(..., description="Text content of the prompt", min_length=1)
    label_id: UUID | None = Field(None, description="Optional label ID to associate with the prompt")

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "TEXT",
                "content": "red car",
                "label_id": "123e4567-e89b-12d3-a456-426614174000",
            }
        }
    }


class VisualPromptCreateSchema(BaseIDPayload):
    """Schema for creating a visual prompt."""

    type: Literal[PromptType.VISUAL]
    frame_id: UUID = Field(..., description="ID of the frame to use for the prompt")
    annotations: list[Annotation] = Field(..., description="List of annotations for the prompt", min_length=1)
    label_id: UUID | None = Field(None, description="Optional label ID to associate with the prompt")

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "VISUAL",
                "frame_id": "123e4567-e89b-12d3-a456-426614174000",
                "annotations": [
                    {
                        "type": "polygon",
                        "points": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
                    }
                ],
                "label_id": "123e4567-e89b-12d3-a456-426614174000",
            }
        }
    }


PromptCreateSchema = Annotated[TextPromptCreateSchema | VisualPromptCreateSchema, Field(discriminator="type")]


class TextPromptUpdateSchema(BaseModel):
    """Schema for updating a text prompt."""

    type: Literal[PromptType.TEXT]
    content: str | None = Field(None, description="Text content of the prompt", min_length=1)
    label_id: UUID | None = Field(None, description="Optional label ID to associate with the prompt")

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "TEXT",
                "content": "red car",
                "label_id": "123e4567-e89b-12d3-a456-426614174000",
            }
        }
    }


class VisualPromptUpdateSchema(BaseModel):
    """Schema for updating a visual prompt."""

    type: Literal[PromptType.VISUAL]
    frame_id: UUID | None = Field(None, description="ID of the frame to use for the prompt")
    annotations: list[Annotation] | None = Field(None, description="List of annotations for the prompt", min_length=1)
    label_id: UUID | None = Field(None, description="Optional label ID to associate with the prompt")

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "VISUAL",
                "frame_id": "123e4567-e89b-12d3-a456-426614174000",
                "annotations": [
                    {
                        "type": "polygon",
                        "points": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
                    }
                ],
                "label_id": "123e4567-e89b-12d3-a456-426614174000",
            }
        }
    }


PromptUpdateSchema = Annotated[TextPromptUpdateSchema | VisualPromptUpdateSchema, Field(discriminator="type")]


class TextPromptSchema(BaseIDSchema):
    """Schema for a text prompt response."""

    type: Literal[PromptType.TEXT]
    content: str


class VisualPromptSchema(BaseIDSchema):
    """Schema for a visual prompt response."""

    type: Literal[PromptType.VISUAL]
    frame_id: UUID
    annotations: list[Annotation]


PromptSchema = Annotated[TextPromptSchema | VisualPromptSchema, Field(discriminator="type")]


class PromptsListSchema(BaseModel):
    """Schema for listing prompts."""

    prompts: list[PromptSchema]
    pagination: Pagination = Field(default_factory=lambda: Pagination(count=0, total=0, offset=0, limit=20))
