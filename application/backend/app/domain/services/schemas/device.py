# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from domain.services.schemas.base import BaseIDSchema, PaginatedResponse


class Device(StrEnum):
    """Enum for configurable types of pipeline components."""

    AUTO = "auto"
    CUDA = "cuda"
    XPU = "xpu"
    CPU = "cpu"


class AvailableDeviceSchema(BaseIDSchema):
    """Single available runtime device."""

    backend: Device
    name: str
    device_id: str


class DevicesListSchema(PaginatedResponse):
    """Wrapper schema for available device list responses."""

    devices: list[AvailableDeviceSchema]
