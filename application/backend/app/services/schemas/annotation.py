# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class AnnotationType(StrEnum):
    POINT = "point"
    POLYGON = "polygon"


class PointAnnotation(BaseModel):
    type: Literal[AnnotationType.POINT]
    x: float = Field(..., description="x coordinate", ge=0.0, le=1.0)
    y: float = Field(..., description="y coordinate", ge=0.0, le=1.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "annotation_type": "point",
                "x": 0.5,
                "y": 0.5,
            }
        }
    }


class PolygonAnnotation(BaseModel):
    type: Literal[AnnotationType.POLYGON]
    points: list[tuple[float, float]]

    model_config = {
        "json_schema_extra": {
            "example": {
                "annotation_type": "polygon",
                "points": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
            }
        }
    }


Annotation = Annotated[PointAnnotation | PolygonAnnotation, Field(discriminator="type")]
