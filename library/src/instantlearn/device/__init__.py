# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cross-runtime device discovery."""

from .device import DeviceInfo, DeviceType, enumerate_system_devices
from .resolver import resolve_device_for_runtime

__all__ = ["DeviceInfo", "DeviceType", "enumerate_system_devices", "resolve_device_for_runtime"]
