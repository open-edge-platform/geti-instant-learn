# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel, Field

from core.components.schemas.processor import ModelConfig
from core.components.schemas.writer import WriterConfig
from services.schemas.base import BaseIDPayload, BaseIDSchema
from services.schemas.source import SourceSchema


class ProjectCreateSchema(BaseIDPayload):
    name: str = Field(max_length=80, min_length=1)


class ProjectUpdateSchema(BaseModel):
    name: str | None = Field(max_length=80, min_length=1, default=None)
    active: bool | None = None


class ProjectSchema(BaseIDSchema):
    name: str


class ProjectsListSchema(BaseModel):
    projects: list[ProjectSchema]


