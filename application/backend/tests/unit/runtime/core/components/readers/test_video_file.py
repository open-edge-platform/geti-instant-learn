#  Copyright (C) 2026 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from domain.services.schemas.reader import ReaderConfig
from runtime.core.components.readers.video_file import VideoFileReader


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.mp4"
    path.write_bytes(b"fake video content")
    return path


@pytest.fixture
def reader_config(video_path: Path) -> MagicMock:
    config = MagicMock(spec=ReaderConfig)
    config.video_path = str(video_path)
    return config


@pytest.fixture
def mock_capture() -> MagicMock:
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        7: 123.0,  # cv2.CAP_PROP_FRAME_COUNT
        5: 25.0,  # cv2.CAP_PROP_FPS
    }.get(prop, 0.0)
    return cap


@pytest.fixture
def reader(reader_config: MagicMock) -> VideoFileReader:
    return VideoFileReader(reader_config)


class TestVideoFileReaderInitialization:
    def test_initialization(self, reader_config: MagicMock) -> None:
        r = VideoFileReader(reader_config)
        assert r._config == reader_config
        assert r._video_capture is None
        assert r._total_frames == 0
        assert r._fps == 30.0
        assert r._video_path is None
        assert r._next_frame_time_s is None


class TestVideoFileReaderConnect:
    def test_connect_success(
        self,
        reader: VideoFileReader,
        video_path: Path,
        mock_capture: MagicMock,
    ) -> None:
        with patch("runtime.core.components.readers.video_file.cv2.VideoCapture", return_value=mock_capture):
            reader.connect()

        assert reader._video_path == video_path
        assert reader._video_capture is mock_capture
        assert reader._total_frames == 123
        assert reader._fps == 25.0
        assert isinstance(reader._next_frame_time_s, float)

    def test_connect_invalid_path_raises(self, tmp_path: Path) -> None:
        config = MagicMock(spec=ReaderConfig)
        config.video_path = str(tmp_path / "missing.mp4")
        r = VideoFileReader(config)

        mock_failed_capture = MagicMock()
        mock_failed_capture.isOpened.return_value = False

        with patch(
            "runtime.core.components.readers.video_file.cv2.VideoCapture",
            return_value=mock_failed_capture,
        ):
            with pytest.raises(RuntimeError, match=r"Failed to open video file:"):
                r.connect()

    def test_connect_open_fails_raises(self, reader: VideoFileReader, mock_capture: MagicMock) -> None:
        mock_capture.isOpened.return_value = False

        with patch("runtime.core.components.readers.video_file.cv2.VideoCapture", return_value=mock_capture):
            with pytest.raises(RuntimeError, match="Failed to open video file"):
                reader.connect()


class TestVideoFileReaderLength:
    def test_len_before_connect(self, reader: VideoFileReader) -> None:
        assert len(reader) == 0

    def test_len_after_connect(self, reader: VideoFileReader, mock_capture: MagicMock) -> None:
        with patch("runtime.core.components.readers.video_file.cv2.VideoCapture", return_value=mock_capture):
            reader.connect()

        assert len(reader) == 123


class TestVideoFileReaderThrottle:
    def test_throttle_sleeps_for_positive_delta(self, reader: VideoFileReader) -> None:
        reader._fps = 10.0
        reader._next_frame_time_s = 101.0

        with (
            patch("runtime.core.components.readers.video_file.time.monotonic", return_value=100.0),
            patch("runtime.core.components.readers.video_file.time.sleep") as mock_sleep,
        ):
            reader._throttle_to_fps()

        mock_sleep.assert_called_once()
        assert mock_sleep.call_args[0][0] == pytest.approx(1.0, abs=1e-6)

    def test_throttle_no_sleep_when_late(self, reader: VideoFileReader) -> None:
        reader._fps = 10.0
        reader._next_frame_time_s = 100.0

        with (
            patch("runtime.core.components.readers.video_file.time.monotonic", return_value=101.0),
            patch("runtime.core.components.readers.video_file.time.sleep") as mock_sleep,
        ):
            reader._throttle_to_fps()

        mock_sleep.assert_not_called()

    def test_throttle_skips_when_fps_non_positive(self, reader: VideoFileReader) -> None:
        reader._fps = 0.0
        reader._next_frame_time_s = 100.0

        with patch("runtime.core.components.readers.video_file.time.sleep") as mock_sleep:
            reader._throttle_to_fps()

        mock_sleep.assert_not_called()


class TestVideoFileReaderRead:
    def test_read_without_connect_returns_none(self, reader: VideoFileReader) -> None:
        assert reader.read() is None

    def test_read_success_returns_inputdata(
        self,
        reader: VideoFileReader,
        mock_capture: MagicMock,
    ) -> None:
        frame_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        frame_rgb = np.ones((10, 10, 3), dtype=np.uint8)

        mock_capture.read.return_value = (True, frame_bgr)

        with (
            patch("runtime.core.components.readers.video_file.cv2.VideoCapture", return_value=mock_capture),
            patch("runtime.core.components.readers.video_file.cv2.cvtColor", return_value=frame_rgb),
            patch("runtime.core.components.readers.video_file.time.sleep"),
            patch("runtime.core.components.readers.video_file.time.monotonic", return_value=100.0),
        ):
            reader.connect()
            data = reader.read()

        assert data is not None
        assert isinstance(data.timestamp, int)
        assert data.frame is frame_rgb
        assert data.context["path"] is not None
        assert data.context["fps"] == 25.0

    def test_read_loops_on_end_of_video(
        self,
        reader: VideoFileReader,
        mock_capture: MagicMock,
    ) -> None:
        frame_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        frame_rgb = np.ones((10, 10, 3), dtype=np.uint8)

        mock_capture.read.side_effect = [
            (False, None),
            (True, frame_bgr),
        ]

        with (
            patch("runtime.core.components.readers.video_file.cv2.VideoCapture", return_value=mock_capture),
            patch("runtime.core.components.readers.video_file.cv2.cvtColor", return_value=frame_rgb),
            patch("runtime.core.components.readers.video_file.time.sleep"),
            patch("runtime.core.components.readers.video_file.time.monotonic", return_value=100.0),
        ):
            reader.connect()
            data = reader.read()

        assert data is not None
        mock_capture.set.assert_called_once()
        args, _kwargs = mock_capture.set.call_args
        assert args[1] == 0

    def test_read_returns_none_if_restart_read_fails(
        self,
        reader: VideoFileReader,
        mock_capture: MagicMock,
    ) -> None:
        mock_capture.read.side_effect = [
            (False, None),
            (False, None),
        ]

        with (
            patch("runtime.core.components.readers.video_file.cv2.VideoCapture", return_value=mock_capture),
            patch("runtime.core.components.readers.video_file.time.sleep"),
            patch("runtime.core.components.readers.video_file.time.monotonic", return_value=100.0),
        ):
            reader.connect()
            data = reader.read()

        assert data is None


class TestVideoFileReaderClose:
    def test_close_releases_capture_and_clears_state(self, reader: VideoFileReader, mock_capture: MagicMock) -> None:
        with patch("runtime.core.components.readers.video_file.cv2.VideoCapture", return_value=mock_capture):
            reader.connect()

        reader.close()

        mock_capture.release.assert_called_once()
        assert reader._video_capture is None
        assert reader._total_frames == 0
        assert reader._video_path is None
        assert reader._next_frame_time_s is None
