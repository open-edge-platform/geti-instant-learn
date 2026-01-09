# test_usb_camera_reader.py

from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from domain.services.schemas.reader import SourceType, UsbCameraConfig
from runtime.core.components.readers.usb_camera_reader import UsbCameraReader


@pytest.fixture
def test_video_path(tmp_path):
    """Create a simple test video file to simulate webcam."""
    video_path = tmp_path / "test_webcam.avi"

    fourcc = cv2.VideoWriter.fourcc(*"MJPG")
    out = cv2.VideoWriter(str(video_path), fourcc, 20.0, (640, 480))

    if not out.isOpened():
        pytest.skip("Could not create test video")

    for i in range(30):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return video_path


@pytest.fixture
def usb_camera_config(test_video_path):
    return UsbCameraConfig(source_type=SourceType.USB_CAMERA, device_id=0)


class TestUsbCameraReader:
    def test_usb_camera_reader_connect(self, usb_camera_config, test_video_path, monkeypatch):
        reader = UsbCameraReader(config=usb_camera_config)
        monkeypatch.setattr(reader._config, "device_id", str(test_video_path))

        reader.connect()

        assert reader._connected is True
        assert reader._cap is not None
        assert reader._cap.isOpened()

        reader.close()

    def test_usb_camera_reader_read_frame(self, usb_camera_config, test_video_path, monkeypatch):
        reader = UsbCameraReader(config=usb_camera_config)
        monkeypatch.setattr(reader._config, "device_id", str(test_video_path))

        reader.connect()

        data = reader.read()

        assert data is not None
        assert data.frame is not None
        assert data.frame.shape == (480, 640, 3)
        assert data.timestamp > 0
        assert isinstance(data.context, dict)

        reader.close()

    def test_usb_camera_reader_end_of_stream(self, usb_camera_config, test_video_path, monkeypatch):
        reader = UsbCameraReader(config=usb_camera_config)
        monkeypatch.setattr(reader._config, "device_id", str(test_video_path))

        reader.connect()

        for _ in range(30):
            data = reader.read()
            assert data is not None

        # Next read should fail
        with pytest.raises(RuntimeError, match="Failed to capture frame"):
            reader.read()

        reader.close()

    def test_usb_camera_reader_connect_invalid_source(self, usb_camera_config, monkeypatch):
        reader = UsbCameraReader(config=usb_camera_config)
        monkeypatch.setattr(reader._config, "device_id", "/nonexistent/video.mp4")

        with pytest.raises(RuntimeError, match="Could not open video source"):
            reader.connect()

    def test_usb_camera_reader_read_before_connect(self, usb_camera_config):
        reader = UsbCameraReader(config=usb_camera_config)

        with pytest.raises(RuntimeError, match="Video capture not initialized"):
            reader.read()


class TestUsbCameraReaderDiscover:
    """Tests for the UsbCameraReader.discover() class method."""

    @pytest.fixture
    def fxt_camera_info_list(self):
        """Mock camera info objects returned by enumerate_cameras."""
        return [
            SimpleNamespace(index=0, name="Integrated Camera", backend="test_backend"),
            SimpleNamespace(index=1, name="USB Webcam", backend="test_backend"),
        ]

    @pytest.mark.parametrize(
        "os_name,backend_count",
        [
            ("Windows", 2),
            ("Darwin", 1),
            ("Linux", 1),
        ],
    )
    def test_discover_uses_correct_backend_for_os(self, os_name, backend_count, fxt_camera_info_list):
        with (
            patch("runtime.core.components.readers.usb_camera_reader.platform.system", return_value=os_name),
            patch("runtime.core.components.readers.usb_camera_reader.enumerate_cameras") as mock_enumerate,
        ):
            mock_enumerate.return_value = fxt_camera_info_list

            result = UsbCameraReader.discover()

            # Verify enumerate_cameras was called for each expected backend
            assert mock_enumerate.call_count == backend_count
            assert len(result) == len(fxt_camera_info_list) * backend_count

    def test_discover_windows_success(self, fxt_camera_info_list):
        with (
            patch("runtime.core.components.readers.usb_camera_reader.platform.system", return_value="Windows"),
            patch("runtime.core.components.readers.usb_camera_reader.enumerate_cameras") as mock_enumerate,
        ):
            mock_enumerate.return_value = fxt_camera_info_list

            result = UsbCameraReader.discover()

            # Windows tries 2 backends, so cameras appear twice
            assert len(result) >= 2
            assert result[0].source_type == SourceType.USB_CAMERA
            assert result[0].device_id == 0
            assert result[0].name == "Integrated Camera"

    def test_discover_macos_success(self, fxt_camera_info_list):
        with (
            patch("runtime.core.components.readers.usb_camera_reader.platform.system", return_value="Darwin"),
            patch("runtime.core.components.readers.usb_camera_reader.enumerate_cameras") as mock_enumerate,
        ):
            mock_enumerate.return_value = fxt_camera_info_list

            result = UsbCameraReader.discover()

            assert len(result) == 2
            assert all(isinstance(cam, UsbCameraConfig) for cam in result)

    def test_discover_linux_success(self, fxt_camera_info_list):
        with (
            patch("runtime.core.components.readers.usb_camera_reader.platform.system", return_value="Linux"),
            patch("runtime.core.components.readers.usb_camera_reader.enumerate_cameras") as mock_enumerate,
        ):
            mock_enumerate.return_value = fxt_camera_info_list

            result = UsbCameraReader.discover()

            assert len(result) == 2
            assert all(isinstance(cam, UsbCameraConfig) for cam in result)

    def test_discover_no_cameras_found(self):
        with (
            patch("runtime.core.components.readers.usb_camera_reader.platform.system", return_value="Linux"),
            patch("runtime.core.components.readers.usb_camera_reader.enumerate_cameras") as mock_enumerate,
        ):
            mock_enumerate.return_value = []

            result = UsbCameraReader.discover()

            assert result == []

    def test_discover_cameras_sorted_by_index(self):
        unsorted_cameras = [
            SimpleNamespace(index=2, name="Camera 2", backend="test"),
            SimpleNamespace(index=0, name="Camera 0", backend="test"),
            SimpleNamespace(index=1, name="Camera 1", backend="test"),
        ]

        with (
            patch("runtime.core.components.readers.usb_camera_reader.platform.system", return_value="Linux"),
            patch("runtime.core.components.readers.usb_camera_reader.enumerate_cameras") as mock_enumerate,
        ):
            mock_enumerate.return_value = unsorted_cameras

            result = UsbCameraReader.discover()

            assert len(result) == 3
            assert result[0].device_id == 0
            assert result[0].name == "Camera 0"
            assert result[1].device_id == 1
            assert result[1].name == "Camera 1"
            assert result[2].device_id == 2
            assert result[2].name == "Camera 2"
