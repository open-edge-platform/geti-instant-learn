# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from services.schemas.base import BaseIDPayload, BaseIDSchema, Pagination

COLOR_REGEX = r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$"


class LabelCreateSchema(BaseIDPayload):
    name: str = Field("", min_length=1, max_length=50, description="Label name")
    color: Annotated[str | None, StringConstraints(pattern=COLOR_REGEX)] = Field(
        None, description="New hex color code, e.g. #RRGGBB or #RGB"
    )


class LabelSchema(BaseIDSchema):
    name: str = Field("", min_length=1, max_length=50, description="Label name")
    color: Annotated[str | None, StringConstraints(pattern=COLOR_REGEX)] = Field(
        None, description="New hex color code, e.g. #RRGGBB or #RGB"
    )


class LabelsListSchema(BaseModel):
    labels: list[LabelSchema]
    pagination: Pagination
