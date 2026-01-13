#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
import platform

import cv2
from cv2_enumerate_cameras import enumerate_cameras

from domain.services.schemas.reader import ReaderConfig, SourceType, UsbCameraConfig
from runtime.core.components.readers.video_stream_reader import BaseOpenCVReader

logger = logging.getLogger(__name__)


class UsbCameraReader(BaseOpenCVReader):
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

    @classmethod
    def discover(cls) -> list[UsbCameraConfig]:
        if platform.system() == "Windows":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        elif platform.system() == "Darwin":  # macOS
            backends = [cv2.CAP_AVFOUNDATION]
        else:  # Linux
            backends = [cv2.CAP_V4L2]

        devices: list[UsbCameraConfig] = []
        camera_list = []

        for backend in backends:
            camera_list.extend(enumerate_cameras(backend))

        camera_list.sort(key=lambda cam: cam.index)

        for camera_info in camera_list:
            devices.append(
                UsbCameraConfig(
                    source_type=SourceType.USB_CAMERA,
                    device_id=camera_info.index,
                    name=camera_info.name,
                )
            )
        logger.info(f"Found {cls.__name__} input device: {devices}")
        return devices
