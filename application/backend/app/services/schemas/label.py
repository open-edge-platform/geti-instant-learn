# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel, Field
from pydantic_extra_types.color import Color

from services.schemas.base import BaseIDPayload, BaseIDSchema, Pagination


class LabelCreateSchema(BaseIDPayload):
    name: str = Field("", min_length=1, max_length=50, description="Label name")
    color: Color | None = Field(None, description="New hex color code, e.g. #RRGGBB")


class LabelSchema(BaseIDSchema):
    name: str = Field("", min_length=1, max_length=50, description="Label name")
    color: str = Field("", description="New hex color code, e.g. #RRGGBB")


class LabelsListSchema(BaseModel):
    labels: list[LabelSchema]
    pagination: Pagination
