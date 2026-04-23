#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging

from domain.services.dataset_discovery import DatasetResolver
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

    def __init__(self, dataset_resolver: DatasetResolver | None = None) -> None:
        """Initialize the factory with a dataset resolver.

        Args:
            dataset_resolver: Service for resolving dataset paths. Required for SampleDatasetConfig.
        """
        self._dataset_resolver = dataset_resolver

    def create(self, config: ReaderConfig | None) -> StreamReader:
        """Create a StreamReader instance based on the provided configuration.

        Args:
            config: Configuration specifying which reader type to create.

        Returns:
            A StreamReader instance configured according to the provided config.

        Raises:
            ValueError: If SampleDatasetConfig is used but no dataset resolver was provided.
        """
        settings = get_settings()
        match config:
            case UsbCameraConfig() as config:
                return UsbCameraReader(config)
            case ImagesFolderConfig() as config:
                return ImageFolderReader(config, supported_extensions=settings.supported_extensions)
            case SampleDatasetConfig() as config:
                if self._dataset_resolver is None:
                    logger.error("DatasetResolver is required for SampleDatasetConfig but was not provided.")
                    raise ValueError("DatasetResolver is required for SampleDatasetConfig.")

                if config.dataset_id is not None:
                    logger.info("Creating sample dataset reader for dataset_id '%s'.", config.dataset_id)
                else:
                    logger.info("Creating sample dataset reader without dataset_id; using first available dataset.")

                dataset_path = self._dataset_resolver.get_dataset_path(dataset_id=config.dataset_id)

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
