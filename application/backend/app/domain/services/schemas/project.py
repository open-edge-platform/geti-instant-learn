# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from pydantic import BaseModel, Field

from domain.services.schemas.base import BaseIDPayload, BaseIDSchema, PaginatedResponse


class Device(StrEnum):
    """Enum for configurable types of pipeline components."""

    AUTO = "auto"
    CUDA = "cuda"
    XPU = "xpu"
    CPU = "cpu"


class ProjectConfig(BaseModel):
    device: Device = Device.CPU

    model_config = {
        "json_schema_extra": {
            "example": {
                "device": "cpu",
            }
        }
    }


class ProjectCreateSchema(BaseIDPayload):
    name: str = Field(max_length=80, min_length=1)
    config: ProjectConfig = Field(default_factory=ProjectConfig)


class ProjectUpdateSchema(BaseModel):
    name: str | None = Field(max_length=80, min_length=1, default=None)
    active: bool | None = None
    config: ProjectConfig | None = None


class ProjectSchema(BaseIDSchema):
    name: str
    active: bool
    config: ProjectConfig


class ProjectsListSchema(PaginatedResponse):
    projects: list[ProjectSchema]
