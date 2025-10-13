# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from services.schemas.base import BaseIDSchema


class SinkSchema(BaseIDSchema):
    config: dict[str, Any]  # TODO update later with strict schema
