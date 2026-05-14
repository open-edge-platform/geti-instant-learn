# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from domain.services.schemas.reader import ImagesFolderConfig, SourceType, VideoFileConfig
from runtime.core.components.validators.reader_config import ReaderConfigValidator


@pytest.fixture
def validator():
    return ReaderConfigValidator()


class TestReaderConfigValidatorVideoFile:
    def test_raises_when_video_file_does_not_exist(self, validator):
        config = VideoFileConfig(source_type=SourceType.VIDEO_FILE, video_path="/nonexistent/video.mp4")

        with pytest.raises(ValueError, match="Video file does not exist: /nonexistent/video.mp4"):
            validator.validate(config)

    def test_raises_when_video_path_is_a_directory(self, validator, tmp_path):
        dir_path = tmp_path / "not_a_video"
        dir_path.mkdir()
        config = VideoFileConfig(source_type=SourceType.VIDEO_FILE, video_path=str(dir_path))

        with pytest.raises(ValueError, match=f"Path is not a file: {dir_path}"):
            validator.validate(config)


class TestReaderConfigValidatorImagesFolder:
    def test_raises_when_images_folder_does_not_exist(self, validator):
        config = ImagesFolderConfig(source_type=SourceType.IMAGES_FOLDER, images_folder_path="/nonexistent/images")

        with pytest.raises(ValueError, match="Images folder does not exist: /nonexistent/images"):
            validator.validate(config)

    def test_raises_when_images_folder_is_a_file(self, validator, tmp_path):
        file_path = tmp_path / "not_a_folder.txt"
        file_path.write_text("test")
        config = ImagesFolderConfig(source_type=SourceType.IMAGES_FOLDER, images_folder_path=str(file_path))

        with pytest.raises(ValueError, match=f"Path is not a directory: {file_path}"):
            validator.validate(config)

    def test_raises_when_images_folder_is_empty(self, validator, tmp_path):
        empty_dir = tmp_path / "empty_folder"
        empty_dir.mkdir()
        config = ImagesFolderConfig(source_type=SourceType.IMAGES_FOLDER, images_folder_path=str(empty_dir))

        with pytest.raises(ValueError, match=f"Images folder is empty: {empty_dir}"):
            validator.validate(config)
