# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field

from domain.services.schemas.base import BaseIDPayload, BaseIDSchema, Pagination
from runtime.core.components.schemas.processor import ModelConfig


class ProcessorSchema(BaseIDSchema):
    config: ModelConfig
    active: bool
    name: str = Field(max_length=80, min_length=1)


class ProcessorListSchema(BaseModel):
    models: list[ProcessorSchema]
    pagination: Pagination


class ProcessorCreateSchema(BaseIDPayload):
    config: ModelConfig
    active: bool
    name: str = Field(max_length=80, min_length=1)


class ProcessorUpdateSchema(BaseModel):
    config: ModelConfig
    active: bool
    name: str = Field(max_length=80, min_length=1)
