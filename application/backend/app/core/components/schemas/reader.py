#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    WEBCAM = "webcam"


class WebCamConfig(BaseModel):
    source_type: Literal[SourceType.WEBCAM]
    id: UUID
    device_id: int
    name: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_type": "webcam",
                "name": "Webcam 0",
                "id": "f9e0ae4f-d96c-4304-baab-2ab845362d03",
                "device_id": 0,
            }
        }
    }


ReaderConfig = Annotated[WebCamConfig, Field(discriminator="source_type")]
