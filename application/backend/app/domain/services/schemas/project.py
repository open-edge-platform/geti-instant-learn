# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from pydantic import BaseModel, Field

from domain.services.schemas.base import BaseIDPayload, BaseIDSchema, PaginatedResponse
from domain.services.schemas.device import Device


class PromptMode(StrEnum):
    """Active prompt mode selected in the UI."""

    TEXT = "text"
    VISUAL = "visual"


class ProjectCreateSchema(BaseIDPayload):
    name: str = Field(max_length=80, min_length=1)
    device: Device = Device.AUTO
    prompt_mode: PromptMode = PromptMode.VISUAL


class ProjectUpdateSchema(BaseModel):
    name: str | None = Field(max_length=80, min_length=1, default=None)
    active: bool | None = None
    device: Device | None = None
    prompt_mode: PromptMode | None = None


class ProjectSchema(BaseIDSchema):
    name: str
    active: bool
    device: Device
    prompt_mode: PromptMode


class ProjectsListSchema(PaginatedResponse):
    projects: list[ProjectSchema]
