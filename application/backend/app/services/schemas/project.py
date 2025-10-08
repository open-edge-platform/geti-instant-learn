# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel, Field

from services.schemas.base import BaseIDPayload, BaseIDSchema
from services.schemas.processor import ProcessorSchema
from services.schemas.sink import SinkSchema
from services.schemas.source import SourceSchema


class ProjectSchema(BaseIDSchema):
    name: str = Field(max_length=80, min_length=1)
    sources: list[SourceSchema] | None = None
    processor: ProcessorSchema | None = None
    sink: SinkSchema | None = None


class ProjectPostPayload(BaseIDPayload):
    name: str = Field(max_length=80, min_length=1)


class ProjectPutPayload(BaseModel):
    name: str = Field(max_length=80, min_length=1)


class ProjectListItem(BaseIDSchema):
    name: str


class ProjectsListSchema(BaseModel):
    projects: list[ProjectListItem]
