# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from pydantic import BaseModel, Field


class Point(BaseModel):
    x: float = Field(..., description="x coordinate", ge=0.0)
    y: float = Field(..., description="y coordinate", ge=0.0)


class PolygonAnnotation(BaseModel):
    points: list[Point] = Field(..., description="Points defining the polygon", min_length=3)

    model_config = {
        "json_schema_extra": {
            "example": {
                "points": [{"x": 1, "y": 1}, {"x": 77, "y": 1}, {"x": 77, "y": 77}, {"x": 1, "y": 77}],
            }
        }
    }


class AnnotationSchema(BaseModel):
    config: PolygonAnnotation
    label_id: UUID = Field(..., description="Label for the annotation")
