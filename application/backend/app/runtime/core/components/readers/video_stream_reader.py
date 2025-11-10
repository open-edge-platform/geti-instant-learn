#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
import time
from abc import ABC, abstractmethod

import cv2

from runtime.core.components.base import StreamReader
from runtime.core.components.schemas.processor import InputData


class BaseOpenCVReader(StreamReader, ABC):
    """Base class for OpenCV-based video readers with common functionality.

    This abstract class provides a template for reading video frames from various
    sources using OpenCV. Subclasses must implement the _connect() method to
    handle source-specific connection logic.
    """

    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._connected = False

    def connect(self) -> None:
        """Connect to the video source.

        Calls the subclass-specific _connect() method and validates the result.

        Raises:
            RuntimeError: If the video source cannot be opened.
        """
        if not self._connected:
            self._cap = self._connect()
            if not self._cap or not self._cap.isOpened():
                raise RuntimeError(f"{self.__class__.__name__}: Could not open video source")
            self._connected = True

    @abstractmethod
    def _connect(self) -> cv2.VideoCapture:
        """Create and configure the OpenCV VideoCapture instance.

        Subclasses must implement this method to provide source-specific
        connection logic (device selection, codec configuration, etc.).

        Returns:
            An opened cv2.VideoCapture instance.

        Raises:
            RuntimeError: If the source cannot be opened.
        """

    def read(self) -> InputData | None:
        if self._cap is None:
            raise RuntimeError(f"{self.__class__.__name__}: Video capture not initialized")

        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError(f"{self.__class__.__name__}: Failed to capture frame")

        # Convert BGR to RGB to conform to the InputData contract
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return InputData(timestamp=int(time.time() * 1000), frame=frame_rgb, context={})

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._connected = False
        super().close()

    def __enter__(self) -> "BaseOpenCVReader":
        self.connect()
        return self
