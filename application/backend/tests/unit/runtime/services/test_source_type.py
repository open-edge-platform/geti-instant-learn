# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from domain.services.schemas.reader import SourceType, UsbCameraConfig
from runtime.services.source_type import SourceTypeService


@pytest.fixture
def service():
    return SourceTypeService()


@pytest.fixture
def fxt_usb_camera_sources():
    return [
        UsbCameraConfig(source_type=SourceType.USB_CAMERA, device_id=0, name="Camera 0"),
        UsbCameraConfig(source_type=SourceType.USB_CAMERA, device_id=1, name="Camera 1"),
    ]


class TestSourceTypeService:
    def test_list_available_sources_usb_camera_success(self, service, fxt_usb_camera_sources):
        with patch("runtime.core.components.readers.usb_camera_reader.UsbCameraReader.discover") as mock_discover:
            mock_discover.return_value = fxt_usb_camera_sources

            result = service.list_available_sources(SourceType.USB_CAMERA)

            mock_discover.assert_called_once()
            assert len(result) == 2
            assert result[0].source_type == SourceType.USB_CAMERA
            assert result[0].device_id == 0
            assert result[0].name == "Camera 0"
            assert result[1].device_id == 1
            assert result[1].name == "Camera 1"

    def test_list_available_sources_usb_camera_empty(self, service):
        with patch("runtime.core.components.readers.usb_camera_reader.UsbCameraReader.discover") as mock_discover:
            mock_discover.return_value = []

            result = service.list_available_sources(SourceType.USB_CAMERA)

            mock_discover.assert_called_once()
            assert result == []

    def test_list_available_sources_not_supported_type(self, service):
        with pytest.raises(ValueError) as exc_info:
            service.list_available_sources(SourceType.VIDEO_FILE)

        assert "Discovery not supported for source type: video_file" in str(exc_info.value)

    def test_list_available_sources_unknown_type(self, service):
        with pytest.raises(ValueError) as exc_info:
            service.list_available_sources("unknown_type")

        assert "Discovery not supported for source type: unknown_type" in str(exc_info.value)
