# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from domain.services.schemas.base import PaginatedResponse
from domain.services.schemas.project import Device


class DevicesListSchema(PaginatedResponse):
    """Wrapper schema for available device list responses."""

    devices: list[Device]
