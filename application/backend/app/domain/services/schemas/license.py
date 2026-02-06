#  Copyright (C) 2026 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel


class LicenseAcceptedSchema(BaseModel):
    """Response schema for license acceptance status."""

    accepted: bool

    model_config = {
        "json_schema_extra": {
            "example": {
                "accepted": True,
            }
        }
    }
