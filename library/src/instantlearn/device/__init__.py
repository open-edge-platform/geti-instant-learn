# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cross-runtime device discovery."""

from .device import DeviceInfo, DeviceType, get_supported_device, get_supported_devices
from .resolver import ResolvedDevice, resolve_device_for_model

__all__ = [
    "DeviceInfo",
    "DeviceType",
    "ResolvedDevice",
    "get_supported_device",
    "get_supported_devices",
    "resolve_device_for_model",
]
