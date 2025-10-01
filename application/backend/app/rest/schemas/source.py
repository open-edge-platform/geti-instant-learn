# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from uuid import UUID

from pydantic import BaseModel

from db.models.source import SourceType


class SourceSchema(BaseModel):
    id: UUID
    type: SourceType
    config: dict[str, Any]  # TODO update later with strict schema
