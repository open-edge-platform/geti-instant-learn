# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field

from domain.db.models import PromptType
from domain.services.schemas.base import BaseIDPayload, BaseIDSchema, PaginatedResponse
from domain.services.schemas.device import Device


class ProjectCreateSchema(BaseIDPayload):
    name: str = Field(max_length=80, min_length=1)
    device: Device = Device.AUTO
    prompt_mode: PromptType = PromptType.VISUAL


class ProjectUpdateSchema(BaseModel):
    name: str | None = Field(max_length=80, min_length=1, default=None)
    active: bool | None = None
    device: Device | None = None
    prompt_mode: PromptType | None = None


class ProjectSchema(BaseIDSchema):
    name: str
    active: bool
    device: Device
    prompt_mode: PromptType


class ProjectsListSchema(PaginatedResponse):
    projects: list[ProjectSchema]
