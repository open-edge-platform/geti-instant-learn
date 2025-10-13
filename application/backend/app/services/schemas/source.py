# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from core.components.schemas.reader import ReaderConfig
from services.schemas.base import BaseIDPayload, BaseIDSchema


class SourceCreateSchema(BaseIDPayload):
    connected: bool
    config: ReaderConfig  # type: ignore[valid-type]


class SourceUpdateSchema(BaseModel):
    connected: bool
    config: ReaderConfig  # type: ignore[valid-type]


class SourceSchema(BaseIDSchema):
    connected: bool
    config: ReaderConfig


class SourcesListSchema(BaseModel):
    sources: list[SourceSchema]
