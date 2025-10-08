# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import Field

from db.models import ProcessorType
from services.schemas.base import BaseIDSchema


class ProcessorSchema(BaseIDSchema):
    type: ProcessorType
    config: dict[str, Any]  # TODO update later with strict schema
    name: str = Field(max_length=80, min_length=1)
