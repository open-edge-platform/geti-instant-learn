#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import cv2

from core.components.readers.video_stream_reader import BaseOpenCVReader
from core.components.schemas.reader import ReaderConfig


class WebCamReader(BaseOpenCVReader):
    """OpenCV-based reader for webcam devices.

    This reader captures video frames from webcam devices using OpenCV.
    It automatically configures the camera with MJPEG codec for cross-platform
    compatibility and optimal performance, especially on Ubuntu 24.04.

    Args:
        config: Reader configuration containing device_id and optional settings.

    Example:
        config = ReaderConfig(device_id=0)  # Use first webcam
        with WebCamReader(config) as reader:
            data = reader.read()
            if data:
                process_frame(data.frame)
    """

    def __init__(self, config: ReaderConfig) -> None:
        """Initialize the webcam reader with configuration.

        Args:
            config: Configuration object containing device_id and optional
                   settings like width, height, and fps.
        """
        self._config = config
        super().__init__()

    def _connect(self) -> cv2.VideoCapture:
        """Connect to the webcam device and configure it.

        Opens the webcam specified by device_id and configures it with:
        - MJPEG codec for cross-platform compatibility
        - Buffer size of 1 to minimize latency

        Returns:
            Configured cv2.VideoCapture instance.

        Raises:
            RuntimeError: If the webcam cannot be opened.
        """
        cap = cv2.VideoCapture(self._config.device_id)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self._config.device_id}")

        # Set MJPEG codec for cross-platform compatibility
        # This prevents freezing issues on Ubuntu 24.04 and provides
        # consistent behavior across Windows, macOS, and Linux
        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),  # type: ignore[attr-defined]
        )

        # Set buffer size to 1 to minimize latency and prevent frame accumulation
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return cap
