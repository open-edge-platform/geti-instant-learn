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


class TestUsbCameraReader:
    def test_usb_camera_reader_connect(self, test_video_path):
        # Create config with integer device id and route capture to test video
        config = UsbCameraConfig(source_type=SourceType.USB_CAMERA, device_id=0, name="Test Camera")
        reader = UsbCameraReader(config=config)

        real_videocapture = cv2.VideoCapture
        with patch("cv2.VideoCapture", side_effect=lambda *_: real_videocapture(str(test_video_path))):
            reader.connect()

        assert reader._connected is True
        assert reader._cap is not None
        assert reader._cap.isOpened()

        reader.close()

    def test_usb_camera_reader_read_frame(self, test_video_path):
        # Create config with integer device id and route capture to test video
        config = UsbCameraConfig(source_type=SourceType.USB_CAMERA, device_id=0, name="Test Camera")
        reader = UsbCameraReader(config=config)

        real_videocapture = cv2.VideoCapture
        with patch("cv2.VideoCapture", side_effect=lambda *_: real_videocapture(str(test_video_path))):
            reader.connect()

        data = reader.read()

        assert data is not None
        assert data.frame is not None
        assert data.frame.shape == (480, 640, 3)
        assert data.timestamp > 0
        assert isinstance(data.context, dict)

        reader.close()

    def test_usb_camera_reader_end_of_stream(self, test_video_path):
        # Create config with integer device id and route capture to test video
        config = UsbCameraConfig(source_type=SourceType.USB_CAMERA, device_id=0, name="Test Camera")
        reader = UsbCameraReader(config=config)

        real_videocapture = cv2.VideoCapture
        with patch("cv2.VideoCapture", side_effect=lambda *_: real_videocapture(str(test_video_path))):
            reader.connect()

        for _ in range(29):
            data = reader.read()
            assert data is not None

        # Next read should fail
        with pytest.raises(RuntimeError, match="Failed to capture frame"):
            reader.read()

        reader.close()

    def test_usb_camera_reader_connect_invalid_source(self):
        # Create config with invalid device id
        config = UsbCameraConfig(
            source_type=SourceType.USB_CAMERA,
            device_id=-1,
            name="Test Camera",
        )
        reader = UsbCameraReader(config=config)

        with patch("cv2.VideoCapture") as mock_cap:
            mock_cap.return_value.isOpened.return_value = False

            with pytest.raises(RuntimeError, match="Could not open video source"):
                reader.connect()

    def test_usb_camera_reader_connect_device_fails_to_capture(self):
        """Test that connection fails if device opens but cannot capture frames."""
        config = UsbCameraConfig(source_type=SourceType.USB_CAMERA, device_id=0, name="Test Camera")
        reader = UsbCameraReader(config=config)

        # Mock a VideoCapture that opens but fails to read
        with patch("cv2.VideoCapture") as mock_cap:
            mock_cap.return_value.isOpened.return_value = True
            mock_cap.return_value.read.return_value = (False, None)
            mock_cap.return_value.set.return_value = True
            mock_cap.return_value.release.return_value = None

            with pytest.raises(RuntimeError, match="opened but failed to capture frames"):
                reader.connect()

    def test_usb_camera_reader_read_before_connect(self):
        config = UsbCameraConfig(source_type=SourceType.USB_CAMERA, device_id=0, name="Test Camera")
        reader = UsbCameraReader(config=config)

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
            assert len(result) == len(fxt_camera_info_list)

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
