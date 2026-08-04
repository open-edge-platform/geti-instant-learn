# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from instantlearn.device import DeviceInfo as LibraryDeviceInfo
from instantlearn.device import DeviceType
from instantlearn.utils.constants import Backend
from pydantic import BaseModel, Field

# Canonical keys use physical classes. XPU/CUDA keys remain accepted as migration aliases.
DEVICE_STR_PATTERN = r"^(?:auto|cpu|(?:gpu|npu|xpu|cuda)(?:-(?:0|[1-9]\d*))?)$"


class DeviceInfo(BaseModel):
    """API representation of a physical runtime device."""

    type: DeviceType = Field(..., description="Physical device type (cpu, gpu, or npu)")
    name: str = Field(..., description="Device name")
    memory: int | None = Field(None, description="Total memory available to the device, in bytes (null for CPU)")
    index: int | None = Field(None, description="Device index among those of the same type (null for CPU)")
    key: str = Field(..., description="Stable device selection key")
    runtime_ids: dict[Backend, str] = Field(..., description="Runtime-specific device identifiers")

    @classmethod
    def from_library(cls, device: LibraryDeviceInfo) -> "DeviceInfo":
        """Build an API schema from the library device registry entry."""
        return cls(
            type=device.type,
            name=device.name,
            memory=device.memory,
            index=device.index,
            key=device.key,
            runtime_ids=device.runtime_ids,
        )
