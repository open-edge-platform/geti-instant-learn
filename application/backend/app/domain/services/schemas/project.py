# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import Literal

from pydantic import BaseModel, Field

from domain.services.schemas.base import BaseIDPayload, BaseIDSchema, PaginatedResponse


class ProjectConfig(BaseModel):
    device: Literal["auto", "cuda", "xpu", "cpu"] = "cpu"

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
