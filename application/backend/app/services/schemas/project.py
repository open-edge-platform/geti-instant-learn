# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel, Field

from services.schemas.base import BaseIDPayload, BaseIDSchema
from services.schemas.source import SourceSchema
from core.components.schemas.reader import ReaderConfig
from core.components.schemas.processor import ModelConfig
from core.components.schemas.writer import WriterConfig


class ProjectCreateSchema(BaseIDPayload):
    name: str = Field(max_length=80, min_length=1)


class ProjectUpdateSchema(BaseModel):
    name: str | None = Field(max_length=80, min_length=1, default=None)
    active: bool | None = None


class ProjectSchema(BaseIDSchema):
    name: str


class ProjectsListSchema(BaseModel):
    projects: list[ProjectSchema]


class ProjectRuntimeConfig(BaseIDSchema):
    name: str
    active: bool
    sources: list[SourceSchema]
    processors: list[ModelConfig]
    sinks: list[WriterConfig]