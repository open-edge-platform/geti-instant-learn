# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from domain.services.schemas.base import BaseIDPayload, BaseIDSchema
from domain.services.schemas.writer import WriterConfig


class SinkCreateSchema(BaseIDPayload):
    connected: bool
    config: WriterConfig


class SinkUpdateSchema(BaseModel):
    connected: bool
    config: WriterConfig


class SinkSchema(BaseIDSchema):
    connected: bool
    config: WriterConfig


class SinksListSchema(BaseModel):
    sinks: list[SinkSchema]
