#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    WEBCAM = "webcam"
    VIDEO_FILE = "video_file"
    IMAGES_FOLDER = "images_folder"


class WebCamConfig(BaseModel):
    source_type: Literal[SourceType.WEBCAM]
    device_id: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_type": "webcam",
                "device_id": 0,
            }
        }
    }


class VideoFileConfig(BaseModel):
    source_type: Literal[SourceType.VIDEO_FILE]
    video_path: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_type": "video_file",
                "video_path": "/path/to/video.mp4",
            }
        }
    }


class ImagesFolderConfig(BaseModel):
    source_type: Literal[SourceType.IMAGES_FOLDER]
    images_folder_path: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_type": "images_folder",
                "images_folder_path": "/path/to/images",
            }
        }
    }


ReaderConfig = Annotated[WebCamConfig | VideoFileConfig | ImagesFolderConfig, Field(discriminator="source_type")]


class FrameMetadata(BaseModel):
    """Metadata for a single frame in the timeline."""

    index: int
    thumbnail: str  # base64-encoded image
    path: str


class FrameListResponse(BaseModel):
    """Paginated response for frame listing."""

    total: int
    page: int
    page_size: int
    frames: list[FrameMetadata]
