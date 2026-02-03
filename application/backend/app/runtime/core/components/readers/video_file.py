#  Copyright (C) 2026 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
import time
from pathlib import Path
from threading import Lock

import cv2

from domain.services.schemas.processor import InputData
from domain.services.schemas.reader import ReaderConfig
from runtime.core.components.base import StreamReader

logger = logging.getLogger(__name__)


class VideoFileReader(StreamReader):
    """Reader implementation for loading frames from a video file.

    This reader opens a video file and reads frames sequentially,
    supporting common video formats (mp4, avi, mov, mkv). When playback
    ends, it automatically restarts from the beginning.

    Args:
        config: Reader configuration specifying the video path and other
            reader-related options.

    Attributes:
        _config: The configuration used to initialize the reader.
        _video_capture: OpenCV video capture handle for the underlying
            video file, or ``None`` if not connected.
        _total_frames: Total number of frames available in the video.
        _fps: Effective frames per second used for throttling reads.
        _video_path: Filesystem path to the video file, or ``None`` if
            not yet connected.
        _next_frame_time_s: Monotonic time (in seconds) at which the
            next frame should be read to respect the target FPS, or
            ``None`` if not scheduled.

    Examples:
        Basic usage::

            from domain.services.schemas.reader import ReaderConfig
            from runtime.core.components.readers.video_file import VideoFileReader

            config = ReaderConfig(video_path="video.mp4")
            reader = VideoFileReader(config)
            reader.connect()
            frame_data = reader.read()
            reader.close()
    """

    def __init__(self, config: ReaderConfig) -> None:
        self._config = config
        self._video_capture: cv2.VideoCapture | None = None
        self._total_frames: int = 0
        self._fps: float = 30.0
        self._video_path: Path | None = None
        self._next_frame_time_s: float | None = None
        self._lock = Lock()
        super().__init__()

    def connect(self) -> None:
        """Open the video file and initialize video properties."""
        video_path = Path(self._config.video_path)
        video_capture = cv2.VideoCapture(str(video_path))

        if not video_capture.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_capture.get(cv2.CAP_PROP_FPS) or 30.0
        next_frame_time_s = time.monotonic()

        with self._lock:
            self._video_path = video_path
            self._video_capture = video_capture
            self._total_frames = total_frames
            self._fps = fps
            self._next_frame_time_s = next_frame_time_s

        logger.info(f"Opened video: {video_path.name}, frames: {total_frames}, fps: {fps:.2f}")

    def __len__(self) -> int:
        """Return the total number of frames in the video."""
        return self._total_frames

    def read(self) -> InputData | None:
        """Read the current frame from the video.

        Returns:
            InputData for the next frame, or None if the reader is closed.

        Notes:
            When the end of the video is reached, playback restarts from the beginning.
        """
        with self._lock:
            # throttle FPS
            if self._fps > 0:
                if self._next_frame_time_s is None:
                    self._next_frame_time_s = time.monotonic()
                else:
                    now_s = time.monotonic()
                    sleep_s = self._next_frame_time_s - now_s
                    if sleep_s > 0:
                        time.sleep(sleep_s)

                    frame_period_s = 1.0 / self._fps
                    self._next_frame_time_s = max(self._next_frame_time_s + frame_period_s, time.monotonic())

            if self._video_capture is None or self._video_path is None:
                return None

            ret, frame = self._video_capture.read()
            if not ret:
                logger.debug("End of video reached, restarting from beginning")
                self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._next_frame_time_s = time.monotonic()

                ret, frame = self._video_capture.read()
                if not ret:
                    logger.error("Failed to read frame after restart")
                    return None

            video_path = self._video_path
            fps = self._fps

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return InputData(
            timestamp=int(time.time() * 1000),
            frame=frame_rgb,
            context={"path": str(video_path), "fps": fps},
        )

    def close(self) -> None:
        """Clean up resources and release video capture."""
        with self._lock:
            if self._video_capture is not None:
                self._video_capture.release()
                self._video_capture = None
            self._total_frames = 0
            self._video_path = None
            self._next_frame_time_s = None
            logger.info("Video reader closed")
