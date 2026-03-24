#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from uuid import UUID, uuid5

from domain.services.schemas.reader import (
    ImagesFolderConfig,
    ReaderConfig,
    SampleDatasetConfig,
    SourceType,
    UsbCameraConfig,
    VideoFileConfig,
)
from runtime.core.components.base import StreamReader
from runtime.core.components.readers.image_folder_reader import ImageFolderReader
from runtime.core.components.readers.noop_reader import NoOpReader
from runtime.core.components.readers.usb_camera_reader import UsbCameraReader
from runtime.core.components.readers.video_file import VideoFileReader
from runtime.errors import DatasetNotFoundError
from runtime.services.dataset_discovery import DATASET_NS
from settings import get_settings

logger = logging.getLogger(__name__)


class StreamReaderFactory:
    """
    A factory for creating StreamReader instances based on a configuration.

    This class decouples the application from the concrete implementation of
    the StreamReader, allowing for different reader types to be instantiated
    based on the provided configuration.
    """

    @staticmethod
    def _resolve_dataset_path(dataset_id: UUID, template_dataset_dir: Path) -> Path | None:
        """Resolve startup-stable dataset ID to a directory path under template datasets."""
        if not template_dataset_dir.exists() or not template_dataset_dir.is_dir():
            return None
        for entry in sorted(template_dataset_dir.iterdir()):
            if entry.is_dir() and uuid5(DATASET_NS, entry.name) == dataset_id:
                return entry
        return None

    @staticmethod
    def _get_first_dataset_path(template_dataset_dir: Path) -> Path | None:
        """Return the first available dataset directory under template datasets."""
        if not template_dataset_dir.exists() or not template_dataset_dir.is_dir():
            return None
        return next((e for e in sorted(template_dataset_dir.iterdir()) if e.is_dir()), None)

    @classmethod
    def create(cls, config: ReaderConfig | None) -> StreamReader:
        settings = get_settings()
        match config:
            case UsbCameraConfig() as config:
                return UsbCameraReader(config)
            case ImagesFolderConfig() as config:
                return ImageFolderReader(config, supported_extensions=settings.supported_extensions)
            case SampleDatasetConfig() as config:
                if config.dataset_id is not None:
                    dataset_path = cls._resolve_dataset_path(config.dataset_id, settings.template_dataset_dir)
                    if dataset_path is None:
                        raise DatasetNotFoundError(f"Sample dataset id '{config.dataset_id}' was not found.")
                else:
                    dataset_path = cls._get_first_dataset_path(settings.template_dataset_dir)
                    if dataset_path is None:
                        raise DatasetNotFoundError("No sample datasets available.")

                template_config = ImagesFolderConfig(
                    source_type=SourceType.IMAGES_FOLDER,
                    images_folder_path=str(dataset_path),
                    seekable=config.seekable,
                )
                return ImageFolderReader(template_config, supported_extensions=settings.supported_extensions)
            case VideoFileConfig() as config:
                return VideoFileReader(config=config)
            case _:
                return NoOpReader()
