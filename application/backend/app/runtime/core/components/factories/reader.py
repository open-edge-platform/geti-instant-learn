#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Mapping
from pathlib import Path
from uuid import UUID

from domain.errors import DatasetNotFoundError
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
from settings import get_settings

logger = logging.getLogger(__name__)


class StreamReaderFactory:
    """
    A factory for creating StreamReader instances based on a configuration.

    This class decouples the application from the concrete implementation of
    the StreamReader, allowing for different reader types to be instantiated
    based on the provided configuration.
    """

    @classmethod
    def create(
        cls,
        config: ReaderConfig | None,
        dataset_paths: Mapping[UUID, Path] | None = None,
    ) -> StreamReader:
        settings = get_settings()
        cached_dataset_paths: Mapping[UUID, Path] = dataset_paths or {}
        match config:
            case UsbCameraConfig() as config:
                return UsbCameraReader(config)
            case ImagesFolderConfig() as config:
                return ImageFolderReader(config, supported_extensions=settings.supported_extensions)
            case SampleDatasetConfig() as config:
                dataset_path: Path | None = None
                if config.dataset_id is not None:
                    logger.info("Creating sample dataset reader for dataset_id '%s'.", config.dataset_id)
                    try:
                        dataset_path = cached_dataset_paths[config.dataset_id]
                    except KeyError as exc:
                        logger.warning(
                            "Sample dataset id '%s' could not be resolved from startup cache.",
                            config.dataset_id,
                        )
                        raise DatasetNotFoundError(f"Sample dataset id '{config.dataset_id}' was not found.") from exc
                else:
                    logger.info("Creating sample dataset reader without dataset_id; using first available dataset.")
                    dataset_path = next(iter(cached_dataset_paths.values()), None)
                    if dataset_path is None:
                        logger.warning("No sample datasets available in startup cache.")
                        raise DatasetNotFoundError("No sample datasets available.")

                logger.info("Using sample dataset path '%s'.", dataset_path)

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
