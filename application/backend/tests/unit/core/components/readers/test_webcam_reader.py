# test_webcam_reader.py

import cv2
import numpy as np
import pytest

from core.components.readers.webcam_reader import WebCamReader
from core.components.schemas.reader import SourceType, WebCamConfig


@pytest.fixture
def test_video_path(tmp_path):
    """Create a simple test video file to simulate webcam."""
    video_path = tmp_path / "test_webcam.avi"

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore[attr-defined]
    out = cv2.VideoWriter(str(video_path), fourcc, 20.0, (640, 480))

    if not out.isOpened():
        pytest.skip("Could not create test video")

    for i in range(30):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return video_path


@pytest.fixture
def webcam_config(test_video_path):
    return WebCamConfig(source_type=SourceType.WEBCAM, device_id=0)


class TestWebCamReader:
    def test_webcam_reader_connect(self, webcam_config, test_video_path, monkeypatch):
        reader = WebCamReader(config=webcam_config)
        monkeypatch.setattr(reader._config, "device_id", str(test_video_path))

        reader.connect()

        assert reader._connected is True
        assert reader._cap is not None
        assert reader._cap.isOpened()

        reader.close()

    def test_webcam_reader_read_frame(self, webcam_config, test_video_path, monkeypatch):
        reader = WebCamReader(config=webcam_config)
        monkeypatch.setattr(reader._config, "device_id", str(test_video_path))

        reader.connect()

        data = reader.read()

        assert data is not None
        assert data.frame is not None
        assert data.frame.shape == (480, 640, 3)
        assert data.timestamp > 0
        assert isinstance(data.context, dict)

        reader.close()

    def test_webcam_reader_end_of_stream(self, webcam_config, test_video_path, monkeypatch):
        reader = WebCamReader(config=webcam_config)
        monkeypatch.setattr(reader._config, "device_id", str(test_video_path))

        reader.connect()

        for _ in range(30):
            data = reader.read()
            assert data is not None

        # Next read should fail
        with pytest.raises(RuntimeError, match="Failed to capture frame"):
            reader.read()

        reader.close()

    def test_webcam_reader_connect_invalid_source(self, webcam_config, monkeypatch):
        reader = WebCamReader(config=webcam_config)
        monkeypatch.setattr(reader._config, "device_id", "/nonexistent/video.mp4")

        with pytest.raises(RuntimeError, match="Could not open video source"):
            reader.connect()

    def test_webcam_reader_read_before_connect(self, webcam_config):
        reader = WebCamReader(config=webcam_config)

        with pytest.raises(RuntimeError, match="Video capture not initialized"):
            reader.read()
