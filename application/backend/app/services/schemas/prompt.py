# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from db.models import PromptType
from services.schemas.annotation import Annotation
from services.schemas.base import BaseIDPayload, BaseIDSchema, Pagination


class TextPromptCreateSchema(BaseIDPayload):
    """Schema for creating a text prompt."""

    type: Literal[PromptType.TEXT]
    name: str = Field(..., description="Name of the prompt", min_length=1, max_length=255)
    content: str = Field(..., description="Text content of the prompt", min_length=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "TEXT",
                "name": "red_car_prompt",
                "content": "red car",
            }
        }
    }


class VisualPromptCreateSchema(BaseIDPayload):
    """Schema for creating a visual prompt."""

    type: Literal[PromptType.VISUAL]
    name: str = Field(..., description="Name of the prompt", min_length=1, max_length=255)
    frame_id: UUID = Field(..., description="ID of the frame to use for the prompt")
    annotations: list[Annotation] = Field(..., description="List of annotations for the prompt", min_length=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "VISUAL",
                "name": "car_bounding_box",
                "frame_id": "123e4567-e89b-12d3-a456-426614174000",
                "annotations": [
                    {
                        "type": "polygon",
                        "points": [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)],
                    }
                ],
            }
        }
    }


PromptCreateSchema = Annotated[TextPromptCreateSchema | VisualPromptCreateSchema, Field(discriminator="type")]


class TextPromptSchema(BaseIDSchema):
    """Schema for a text prompt response."""

    type: Literal[PromptType.TEXT]
    name: str
    content: str


class VisualPromptSchema(BaseIDSchema):
    """Schema for a visual prompt response."""

    type: Literal[PromptType.VISUAL]
    name: str
    frame_id: UUID
    annotations: list[Annotation]


PromptSchema = Annotated[TextPromptSchema | VisualPromptSchema, Field(discriminator="type")]


class PromptsListSchema(BaseModel):
    """Schema for listing prompts."""
    prompts: list[PromptSchema]
    pagination: Pagination

