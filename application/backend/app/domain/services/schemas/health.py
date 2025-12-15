#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class HealthStatus(StrEnum):
    OK = "ok"


class HealthCheckSchema(BaseModel):
    status: Literal[HealthStatus.OK]

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
            }
        }
    }
