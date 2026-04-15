# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel, Field
from pydantic_extra_types.color import Color

from domain.services.schemas.base import BaseIDPayload, BaseIDSchema, PaginatedResponse


class LabelCreateSchema(BaseIDPayload):
    name: str = Field("", min_length=1, max_length=50, description="Label name")
    color: Color | None = Field(None, description="New hex color code, e.g. #RRGGBB")


class LabelSchema(BaseIDSchema):
    name: str = Field("", min_length=1, max_length=50, description="Label name")
    color: str = Field("", description="New hex color code, e.g. #RRGGBB")


class LabelUpdateSchema(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=50, description="Label name")
    color: Color | None = Field(None, description="New hex color code, e.g. #RRGGBB")


class LabelsListSchema(PaginatedResponse):
    labels: list[LabelSchema]


@dataclass(frozen=True)
class RGBColor:
    """Immutable RGB color representation."""

    r: int
    g: int
    b: int

    def to_tuple(self) -> tuple[int, int, int]:
        return self.r, self.g, self.b


@dataclass(frozen=True)
class VisualizationLabel:
    """Label data needed for visualization."""

    id: UUID
    color: RGBColor
    object_name: str | None = None


@dataclass(frozen=True)
class CategoryMappings:
    """Bidirectional category ID mappings."""

    label_to_category_id: dict[UUID, int]
    category_id_to_label_id: dict[int, str]


@dataclass(frozen=True)
class VisualizationInfo:
    """Complete visualization metadata for a pipeline."""

    label_colors: list[VisualizationLabel]
    category_mappings: CategoryMappings
