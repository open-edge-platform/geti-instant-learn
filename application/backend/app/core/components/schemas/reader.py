#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    WEBCAM = "webcam"
    IP_CAMERA = "ip_camera"
    VIDEO_FILE = "video_file"
    IMAGES_FOLDER = "images_folder"


class WebCamConfig(BaseModel):
    source_type: Literal[SourceType.WEBCAM]
    device_id: int
    name: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_type": "webcam",
                "name": "Webcam 0",
                "device_id": 0,
            }
        }
    }


class IPCameraConfig(BaseModel):
    source_type: Literal[SourceType.IP_CAMERA]
    stream_url: str
    auth_required: bool = False
    name: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_type": "ip_camera",
                "name": "Street Camera 123",
                "stream_url": "http://example.com/stream",
                "auth_required": True,
            }
        }
    }


ReaderConfig = Annotated[WebCamConfig | IPCameraConfig, Field(discriminator="source_type")]
