#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
import platform
import time

import cv2
from cv2_enumerate_cameras import enumerate_cameras

from domain.services.schemas.reader import ReaderConfig, SourceType, UsbCameraConfig
from runtime.core.components.readers.video_stream_reader import BaseOpenCVReader

logger = logging.getLogger(__name__)


class UsbCameraReader(BaseOpenCVReader):
    _initial_frame_attempts = 60
    _initial_frame_retry_interval_sec = 0.1

    def __init__(self, config: ReaderConfig) -> None:
        self._config = config
        super().__init__()

    def _connect(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._config.device_id)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self._config.device_id}")

        # Try to set MJPEG codec for cross-platform compatibility
        # Some cameras may not support this, so we don't fail if it doesn't work
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
        except Exception as e:
            logger.warning(f"Could not set MJPEG codec for device {self._config.device_id}: {e}")

        # Validate the camera by reading an initial frame.
        # Right after source switches, some cameras need a short warm-up period.
        frame = None
        for attempt in range(self._initial_frame_attempts):
            ret, frame = cap.read()
            if ret and frame is not None:
                break

            if attempt < self._initial_frame_attempts - 1:
                time.sleep(self._initial_frame_retry_interval_sec)

        if frame is None:
            cap.release()
            raise RuntimeError(
                f"Camera {self._config.name} on device {self._config.device_id} opened but failed to capture frames. "
                "The device may not be a valid capture device or may be in use by another application."
            )

        logger.info(
            f"Successfully connected to USB camera {self._config.name} from device {self._config.device_id},"
            f" frame shape: {frame.shape}"
        )
        return cap

    @classmethod
    def discover(cls) -> list[UsbCameraConfig]:
        if platform.system() == "Windows":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        elif platform.system() == "Darwin":  # macOS
            backends = [cv2.CAP_AVFOUNDATION]
        else:  # Linux
            backends = [cv2.CAP_V4L2]

        devices: set[UsbCameraConfig] = set()

        for backend in backends:
            for camera_info in enumerate_cameras(backend):
                devices.add(
                    UsbCameraConfig(
                        source_type=SourceType.USB_CAMERA,
                        device_id=camera_info.index,
                        name=camera_info.name,
                    )
                )

        sorted_devices = sorted(devices, key=lambda device: device.device_id)
        logger.info(f"Found {len(sorted_devices)} {cls.__name__} device(s): {sorted_devices}")
        return sorted_devices
