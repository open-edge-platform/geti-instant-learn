# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from db.models.processor import ProcessorType


class ProcessorSchema(BaseModel):
    id: UUID
    type: ProcessorType
    config: dict[str, Any]  # TODO update later with strict schema
    name: str = Field(max_length=80, min_length=1)
