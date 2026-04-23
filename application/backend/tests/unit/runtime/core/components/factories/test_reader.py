#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from domain.errors import DatasetNotFoundError
from domain.services.dataset_discovery import DatasetResolver
from domain.services.schemas.reader import (
    ImagesFolderConfig,
    SampleDatasetConfig,
    SourceType,
    UsbCameraConfig,
    VideoFileConfig,
)
from runtime.core.components.factories.reader import StreamReaderFactory
from runtime.core.components.readers.image_folder_reader import ImageFolderReader
from runtime.core.components.readers.noop_reader import NoOpReader
from runtime.core.components.readers.usb_camera_reader import UsbCameraReader
from runtime.core.components.readers.video_file import VideoFileReader


class TestStreamReaderFactory:
    def test_factory_returns_usb_camera_reader(self):
        usb_camera_config = UsbCameraConfig(source_type=SourceType.USB_CAMERA, device_id=1)
        factory = StreamReaderFactory()

        result = factory.create(usb_camera_config)

        assert isinstance(result, UsbCameraReader)
        assert result._config == usb_camera_config

    def test_factory_returns_noop_reader_for_other_config(self):
        factory = StreamReaderFactory()

        result = factory.create(None)

        assert isinstance(result, NoOpReader)

    def test_factory_returns_image_folder_reader_for_images_folder_config(self, tmp_path):
        image_file = tmp_path / "test.jpg"
        image_file.touch()

        config = ImagesFolderConfig(
            source_type=SourceType.IMAGES_FOLDER,
            images_folder_path=str(tmp_path),
        )
        factory = StreamReaderFactory()

        result = factory.create(config)

        assert isinstance(result, ImageFolderReader)
        assert result._config == config

    def test_factory_returns_video_file_reader_for_video_file_config(self, tmp_path: Path) -> None:
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video content")

        config = VideoFileConfig(source_type=SourceType.VIDEO_FILE, video_path=str(video_path))
        factory = StreamReaderFactory()

        result = factory.create(config)

        assert isinstance(result, VideoFileReader)
        assert result._config == config

    def test_factory_returns_sample_dataset_reader_for_cached_dataset_id(self, tmp_path: Path, monkeypatch) -> None:
        dataset_dir = tmp_path / "aquarium"
        dataset_dir.mkdir()
        (dataset_dir / "frame.jpg").touch()
        dataset_id = uuid4()
        config = SampleDatasetConfig(source_type=SourceType.SAMPLE_DATASET, dataset_id=dataset_id)
        monkeypatch.setattr(
            "runtime.core.components.factories.reader.get_settings",
            lambda: SimpleNamespace(supported_extensions={".jpg", ".png"}),
        )

        dataset_resolver = Mock(spec=DatasetResolver)
        dataset_resolver.get_dataset_path.return_value = dataset_dir
        factory = StreamReaderFactory(dataset_resolver=dataset_resolver)

        result = factory.create(config)

        assert isinstance(result, ImageFolderReader)
        assert result._config.images_folder_path == str(dataset_dir)
        dataset_resolver.get_dataset_path.assert_called_once_with(dataset_id=dataset_id)

    def test_factory_raises_for_unknown_sample_dataset_id(self, monkeypatch) -> None:
        dataset_id = uuid4()
        config = SampleDatasetConfig(source_type=SourceType.SAMPLE_DATASET, dataset_id=dataset_id)
        monkeypatch.setattr(
            "runtime.core.components.factories.reader.get_settings",
            lambda: SimpleNamespace(supported_extensions={".jpg", ".png"}),
        )

        dataset_resolver = Mock(spec=DatasetResolver)
        dataset_resolver.get_dataset_path.side_effect = DatasetNotFoundError(f"Dataset {dataset_id} not found")
        factory = StreamReaderFactory(dataset_resolver=dataset_resolver)

        with pytest.raises(DatasetNotFoundError, match="not found"):
            factory.create(config)

    def test_factory_uses_first_cached_sample_dataset_when_dataset_id_missing(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        first_dataset_dir = tmp_path / "aquarium"
        first_dataset_dir.mkdir()
        (first_dataset_dir / "frame.jpg").touch()
        config = SampleDatasetConfig(source_type=SourceType.SAMPLE_DATASET, dataset_id=None)
        monkeypatch.setattr(
            "runtime.core.components.factories.reader.get_settings",
            lambda: SimpleNamespace(supported_extensions={".jpg", ".png"}),
        )

        dataset_resolver = Mock(spec=DatasetResolver)
        dataset_resolver.get_dataset_path.return_value = first_dataset_dir
        factory = StreamReaderFactory(dataset_resolver=dataset_resolver)

        result = factory.create(config)

        assert isinstance(result, ImageFolderReader)
        assert result._config.images_folder_path == str(first_dataset_dir)
        dataset_resolver.get_dataset_path.assert_called_once_with(dataset_id=None)

    def test_factory_raises_when_no_cached_sample_datasets_are_available(self, monkeypatch) -> None:
        config = SampleDatasetConfig(source_type=SourceType.SAMPLE_DATASET, dataset_id=None)
        monkeypatch.setattr(
            "runtime.core.components.factories.reader.get_settings",
            lambda: SimpleNamespace(supported_extensions={".jpg", ".png"}),
        )

        dataset_resolver = Mock(spec=DatasetResolver)
        dataset_resolver.get_dataset_path.side_effect = DatasetNotFoundError("No sample datasets available")
        factory = StreamReaderFactory(dataset_resolver=dataset_resolver)

        with pytest.raises(DatasetNotFoundError, match="No sample datasets available"):
            factory.create(config)

    def test_factory_raises_when_dataset_resolver_not_provided_for_sample_dataset(self, monkeypatch) -> None:
        config = SampleDatasetConfig(source_type=SourceType.SAMPLE_DATASET, dataset_id=None)
        monkeypatch.setattr(
            "runtime.core.components.factories.reader.get_settings",
            lambda: SimpleNamespace(supported_extensions={".jpg", ".png"}),
        )
        factory = StreamReaderFactory(dataset_resolver=None)

        with pytest.raises(ValueError, match="DatasetResolver is required"):
            factory.create(config)


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
