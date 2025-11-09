#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import cv2

from runtime.core.components.readers.video_stream_reader import BaseOpenCVReader
from runtime.core.components.schemas.reader import ReaderConfig


class WebCamReader(BaseOpenCVReader):
    def __init__(self, config: ReaderConfig) -> None:
        self._config = config
        super().__init__()

    def _connect(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._config.device_id)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self._config.device_id}")

        # Set MJPEG codec for cross-platform compatibility
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))

        return cap
