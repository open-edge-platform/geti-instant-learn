# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from domain.services.schemas.base import BaseIDSchema, PaginatedResponse
from domain.services.schemas.project import Device


class AvailableDeviceSchema(BaseIDSchema):
    """Single available runtime device."""

    backend: Device
    name: str


class DevicesListSchema(PaginatedResponse):
    """Wrapper schema for available device list responses."""

    devices: list[AvailableDeviceSchema]
