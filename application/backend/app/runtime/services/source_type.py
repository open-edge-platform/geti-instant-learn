# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from domain.services.schemas.reader import ReaderConfig, SourceType
from runtime.core.components.base import StreamReader
from runtime.core.components.readers.usb_camera_reader import UsbCameraReader


class SourceTypeService:
    def __init__(self) -> None:
        self._discoverable_sources: dict[str, type[StreamReader]] = {
            SourceType.USB_CAMERA: UsbCameraReader,
        }

    def list_available_sources(self, source_type: str) -> list[ReaderConfig]:
        if source_type not in self._discoverable_sources:
            raise ValueError(f"Discovery not supported for source type: {source_type}")

        return self._discoverable_sources[source_type].discover()
