# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from domain.services.schemas.base import PaginatedResponse
from domain.services.schemas.project import Device


class AvailableDeviceSchema(BaseModel):
    """Single available runtime device."""

    backend: Device
    device_id: str
    name: str
    index: int | None = None


class DevicesListSchema(PaginatedResponse):
    """Wrapper schema for available device list responses."""

    devices: list[AvailableDeviceSchema]
