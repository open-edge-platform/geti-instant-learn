# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from domain.services.schemas.base import BaseIDPayload, BaseIDSchema, PaginatedResponse
from domain.services.schemas.reader import ReaderConfig


class SourceCreateSchema(BaseIDPayload):
    connected: bool
    config: ReaderConfig  # type: ignore[valid-type]


class SourceUpdateSchema(BaseModel):
    connected: bool
    config: ReaderConfig  # type: ignore[valid-type]


class SourceSchema(BaseIDSchema):
    connected: bool
    config: ReaderConfig


class SourcesListSchema(PaginatedResponse):
    sources: list[SourceSchema]
