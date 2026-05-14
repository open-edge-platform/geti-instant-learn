#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from domain.services.schemas.base import PaginatedResponse


class SourceType(StrEnum):
    USB_CAMERA = "usb_camera"
    VIDEO_FILE = "video_file"
    IMAGES_FOLDER = "images_folder"
    SAMPLE_DATASET = "sample_dataset"


class UsbCameraConfig(BaseModel):
    source_type: Literal[SourceType.USB_CAMERA]
    name: str | None = None
    device_id: int
    seekable: bool = False

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Optional name",
                "seekable": False,
                "source_type": "usb_camera",
                "device_id": 0,
            }
        }
    }


class VideoFileConfig(BaseModel):
    source_type: Literal[SourceType.VIDEO_FILE]
    video_path: str
    seekable: bool = True

    model_config = {
        "json_schema_extra": {
            "example": {
                "seekable": True,
                "source_type": "video_file",
                "video_path": "/path/to/video.mp4",
            }
        }
    }


class ImagesFolderConfig(BaseModel):
    source_type: Literal[SourceType.IMAGES_FOLDER]
    images_folder_path: str
    seekable: bool = True

    model_config = {
        "json_schema_extra": {
            "example": {
                "seekable": True,
                "source_type": "images_folder",
                "images_folder_path": "/path/to/images",
            }
        }
    }


class SampleDatasetConfig(BaseModel):
    """Configuration for using the pre-configured template dataset.

    The actual dataset path is resolved from application settings at the factory level,
    making this config UI-agnostic.
    """

    source_type: Literal[SourceType.SAMPLE_DATASET]
    seekable: bool = True
    dataset_id: UUID | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "seekable": True,
                "source_type": "sample_dataset",
                "dataset_id": "a12de9d4-5f52-5f1f-a819-fe7c5186dc53",
            }
        }
    }


ReaderConfig = Annotated[
    UsbCameraConfig | VideoFileConfig | ImagesFolderConfig | SampleDatasetConfig,
    Field(discriminator="source_type"),
]


class FrameMetadata(BaseModel):
    """Metadata for a single frame in the timeline."""

    index: int
    thumbnail: str


class FrameListResponse(PaginatedResponse):
    """Paginated response for frame listing."""

    frames: list[FrameMetadata]


class FrameIndexResponse(BaseModel):
    """Response for current frame index."""

    index: int
