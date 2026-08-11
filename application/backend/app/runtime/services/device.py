# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import re

from instantlearn.device import DeviceInfo, DeviceType, get_supported_devices

from domain.services.schemas.device import DEVICE_STR_PATTERN

logger = logging.getLogger(__name__)

_DEVICE_RE = re.compile(DEVICE_STR_PATTERN)


class DeviceService:
    """Single source of truth for available devices and device-string resolution.

    The list of devices is captured at construction time (typically at app startup) and used
    for both listing (API) and resolution (model factory).
    """

    def __init__(self, devices: list[DeviceInfo]) -> None:
        self._devices = list(devices)

    @classmethod
    def from_system(cls) -> "DeviceService":
        """Construct a service from a fresh enumeration of the local system."""
        return cls(devices=get_supported_devices())

    def list_devices(self) -> list[DeviceInfo]:
        """Return the cached list of real available devices."""
        return list(self._devices)

    @staticmethod
    def parse(device_str: str) -> tuple[str, int | None]:
        """Parse a device selection key into ``(kind, index)``.

        Raises:
            ValueError: When the string doesn't match the canonical format.
        """
        normalized = device_str.lower()
        if not _DEVICE_RE.match(normalized):
            raise ValueError(
                f"Invalid device string: {device_str!r}. Expected 'auto', 'cpu', '<gpu|npu>', or '<gpu|npu>-<N>'."
            )
        if "-" in normalized:
            type_str, idx_str = normalized.split("-", 1)
            return type_str, int(idx_str)
        return normalized, None

    def _find_device(self, kind: str, index: int | None) -> DeviceInfo | None:
        if kind == DeviceType.CPU.value:
            return next((device for device in self._devices if device.type == DeviceType.CPU), None)

        target_index = 0 if index is None else index
        device_type = DeviceType(kind)
        return next(
            (device for device in self._devices if device.type == device_type and device.index == target_index),
            None,
        )

    def validate(self, device_str: str) -> bool:
        """Return True if the device string is syntactically valid AND currently available."""
        try:
            kind, index = self.parse(device_str)
        except ValueError:
            return False
        if kind == "auto":
            return True
        return self._find_device(kind, index) is not None

    def resolve_preference(self, device_str: str) -> DeviceInfo | None:
        """Resolve a stored preference to a device, or ``None`` for auto-selection.

        Unavailable and unparsable values fall back to automatic selection.
        """
        try:
            kind, index = self.parse(device_str)
        except ValueError:
            logger.warning("Invalid device string %r; falling back to auto.", device_str)
            return None

        if kind == "auto":
            return None

        match = self._find_device(kind, index)
        if match is None:
            logger.warning("Configured device %r is not available on this system; falling back to auto.", device_str)
            return None
        return match
