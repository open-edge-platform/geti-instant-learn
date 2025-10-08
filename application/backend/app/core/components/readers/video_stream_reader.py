#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
import time
from abc import ABC

import cv2

from core.components.base import StreamReader
from core.components.schemas.processor import InputData
from core.components.schemas.reader import ReaderConfig


class BaseOpenCVReader(StreamReader, ABC):
    """Base class for OpenCV-based video reader with common functionality."""

    def __init__(self, source: str | int, config: ReaderConfig) -> None:
        """Initialize OpenCV reader.
        Args:
            source: Video source (device ID, file path, or URL)
            source_type: Type of the video source
            **config: Additional metadata for the reader
        """
        self.source = source
        self.source_type = config.source_type
        self.config = config
        self._cap: cv2.VideoCapture | None = None
        self.connected = False

    def connect(self) -> None:
        if not self.connected:
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open video source: {self.source}")
            self.connected = True

    def read(self) -> InputData | None:
        """Read a frame from the capture device."""
        if self._cap is None:
            raise RuntimeError("Video capture not initialized")

        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError(f"Failed to capture frame from {self.source_type.value}")

        return InputData(timestamp=int(time.time() * 1000), frame=frame, context={})

    def close(self) -> None:
        """Release OpenCV VideoCapture resources."""
        if self._cap is not None:
            self._cap.release()
        super().close()
