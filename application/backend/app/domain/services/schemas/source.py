# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from domain.services.schemas.base import BaseIDPayload, BaseIDSchema, PaginatedResponse
from domain.services.schemas.reader import ReaderConfig


class SourceCreateSchema(BaseIDPayload):
    active: bool
    config: ReaderConfig  # type: ignore[valid-type]


class SourceUpdateSchema(BaseModel):
    active: bool
    config: ReaderConfig  # type: ignore[valid-type]


class SourceSchema(BaseIDSchema):
    active: bool
    config: ReaderConfig


class SourcesListSchema(PaginatedResponse):
    sources: list[SourceSchema]
