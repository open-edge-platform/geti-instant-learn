# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class AnnotationType(StrEnum):
    RECTANGLE = "rectangle"
    POLYGON = "polygon"


class Point(BaseModel):
    x: float = Field(..., description="x coordinate", ge=0.0, le=1.0)
    y: float = Field(..., description="y coordinate", ge=0.0, le=1.0)


class RectangleAnnotation(BaseModel):
    type: Literal[AnnotationType.RECTANGLE]
    points: list[Point] = Field(
        ..., description="Two points defining rectangle: top-left and bottom-right", min_length=2, max_length=2
    )

    @field_validator("points")
    @classmethod
    def validate_rectangle_order(cls, v: list[Point]) -> list[Point]:
        if v[0].x >= v[1].x or v[0].y >= v[1].y:
            raise ValueError("First point must be top-left, second point must be bottom-right (x1 < x2 and y1 < y2)")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "rectangle",
                "points": [{"x": 0.1, "y": 0.1}, {"x": 0.5, "y": 0.5}],
            }
        }
    }


class PolygonAnnotation(BaseModel):
    type: Literal[AnnotationType.POLYGON]
    points: list[Point] = Field(..., description="Points defining the polygon", min_length=3)

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "polygon",
                "points": [{"x": 0.1, "y": 0.1}, {"x": 0.5, "y": 0.1}, {"x": 0.5, "y": 0.5}, {"x": 0.1, "y": 0.5}],
            }
        }
    }


Annotation = Annotated[RectangleAnnotation | PolygonAnnotation, Field(discriminator="type")]


class AnnotationSchema(BaseModel):
    config: Annotation
    label_id: UUID | None = Field(None, description="Optional label for the annotation")
