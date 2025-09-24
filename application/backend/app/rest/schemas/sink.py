# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from uuid import UUID

from pydantic import BaseModel


class SinkSchema(BaseModel):
    id: UUID
    config: dict[str, Any]  # TODO update later with strict schema
