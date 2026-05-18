# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field

from domain.db.models import PromptType
from domain.services.schemas.base import BaseIDPayload, BaseIDSchema, PaginatedResponse
from domain.services.schemas.device import DEVICE_STR_PATTERN


class ProjectCreateSchema(BaseIDPayload):
    name: str = Field(max_length=80, min_length=1)
    device: str = Field(default="auto", pattern=DEVICE_STR_PATTERN)
    prompt_mode: PromptType = PromptType.VISUAL


class ProjectUpdateSchema(BaseModel):
    name: str | None = Field(max_length=80, min_length=1, default=None)
    active: bool | None = None
    device: str | None = Field(default=None, pattern=DEVICE_STR_PATTERN)
    prompt_mode: PromptType | None = None


class ProjectSchema(BaseIDSchema):
    name: str
    active: bool
    device: str = Field(pattern=DEVICE_STR_PATTERN)
    prompt_mode: PromptType


class ProjectsListSchema(PaginatedResponse):
    projects: list[ProjectSchema]
