#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from domain.services.schemas.reader import (
    ImagesFolderConfig,
    SampleDatasetConfig,
    SourceType,
    VideoFileConfig,
    WebCamConfig,
)
from runtime.core.components.factories.reader import StreamReaderFactory
from runtime.core.components.readers.image_folder_reader import ImageFolderReader
from runtime.core.components.readers.noop_reader import NoOpReader
from runtime.core.components.readers.video_file import VideoFileReader
from runtime.core.components.readers.webcam_reader import WebCamReader


class TestStreamReaderFactory:
    def test_factory_returns_webcam_reader(self):
        webcam_config = WebCamConfig(source_type=SourceType.WEBCAM, device_id=1)

        result = StreamReaderFactory.create(webcam_config)

        assert isinstance(result, WebCamReader)
        assert result._config == webcam_config

    def test_factory_returns_noop_reader_for_other_config(self):
        result = StreamReaderFactory.create(None)

        assert isinstance(result, NoOpReader)

    def test_factory_returns_image_folder_reader_for_images_folder_config(self, tmp_path):
        image_file = tmp_path / "test.jpg"
        image_file.touch()

        config = ImagesFolderConfig(
            source_type=SourceType.IMAGES_FOLDER,
            images_folder_path=str(tmp_path),
        )

        result = StreamReaderFactory.create(config)

        assert isinstance(result, ImageFolderReader)
        assert result._config == config

    def test_factory_returns_image_folder_reader_for_template_dataset_config(self, tmp_path):
        image_file = tmp_path / "test.jpg"
        image_file.touch()

        config = SampleDatasetConfig(source_type=SourceType.SAMPLE_DATASET)

        with patch("runtime.core.components.factories.reader.get_settings") as mock_settings:
            mock_settings.return_value.template_dataset_dir = tmp_path
            mock_settings.return_value.supported_extensions = {".jpg", ".jpeg", ".png"}
            result = StreamReaderFactory.create(config)

        assert isinstance(result, ImageFolderReader)
        assert isinstance(result._config, ImagesFolderConfig)
        assert result._config.images_folder_path == str(tmp_path)

    def test_factory_returns_video_file_reader_for_video_file_config(self, tmp_path: Path) -> None:
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video content")

        config = VideoFileConfig(source_type=SourceType.VIDEO_FILE, video_path=str(video_path))

        result = StreamReaderFactory.create(config)

        assert isinstance(result, VideoFileReader)
        assert result._config == config


class TestImagesFolderConfigValidation:
    @pytest.mark.parametrize(
        "path_setup,error_match",
        [
            pytest.param(
                lambda tmp_path: "/nonexistent/path/to/images",
                "Images folder does not exist",
                id="path_does_not_exist",
            ),
            pytest.param(
                lambda tmp_path: str((tmp_path / "file.txt").absolute())
                if (tmp_path / "file.txt").touch() or True
                else "",
                "Path is not a directory",
                id="path_is_not_directory",
            ),
            pytest.param(
                lambda tmp_path: str(tmp_path),
                "Images folder is empty",
                id="folder_is_empty",
            ),
        ],
    )
    def test_images_folder_config_validation_fails(
        self,
        tmp_path: Path,
        path_setup: Callable[[Path], str],
        error_match: str,
    ) -> None:
        path = path_setup(tmp_path)

        with pytest.raises(ValueError, match=error_match):
            ImagesFolderConfig(source_type=SourceType.IMAGES_FOLDER, images_folder_path=path)
