# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import re

from domain.services.schemas.device import DEVICE_STR_PATTERN, DeviceInfo, DeviceType

logger = logging.getLogger(__name__)

_DEVICE_RE = re.compile(DEVICE_STR_PATTERN)


def _list_xpu_devices() -> list[DeviceInfo]:
    """List all Intel XPU devices exposed by PyTorch."""
    try:
        import torch

        if not torch.xpu.is_available():
            return []

        devices: list[DeviceInfo] = []
        for index in range(torch.xpu.device_count()):
            name = torch.xpu.get_device_name(index)
            props = torch.xpu.get_device_properties(index)
            memory = getattr(props, "total_memory", None)
            devices.append(DeviceInfo(type=DeviceType.XPU, name=name, memory=memory, index=index))
        return devices
    except (ImportError, AttributeError, RuntimeError):
        return []


def _list_cuda_devices() -> list[DeviceInfo]:
    """List all CUDA devices exposed by PyTorch."""
    try:
        import torch

        if not torch.cuda.is_available():
            return []

        devices: list[DeviceInfo] = []
        for index in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(index)
            props = torch.cuda.get_device_properties(index)
            memory = getattr(props, "total_memory", None)
            devices.append(DeviceInfo(type=DeviceType.CUDA, name=name, memory=memory, index=index))
        return devices
    except (ImportError, AttributeError, RuntimeError):
        return []


def _cpu_info() -> DeviceInfo:
    return DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)


def _auto_info() -> DeviceInfo:
    return DeviceInfo(type=DeviceType.AUTO, name="AUTO", memory=None, index=None)


def enumerate_system_devices() -> list[DeviceInfo]:
    """Enumerate all currently available runtime devices.

    CPU is always present. Intel XPU and NVIDIA CUDA devices are enumerated when detected.
    """
    return [*_list_xpu_devices(), *_list_cuda_devices(), _cpu_info()]


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
        return cls(devices=enumerate_system_devices())

    def list_devices(self) -> list[DeviceInfo]:
        """Return the cached list of real available devices."""
        return list(self._devices)

    @staticmethod
    def parse(device_str: str) -> tuple[DeviceType, int | None]:
        """Parse a ``<type>[-<index>]`` string into ``(type, index)``.

        Raises:
            ValueError: When the string doesn't match the canonical format or carries
                an index for a type that doesn't support one.
        """
        normalized = device_str.lower()
        if not _DEVICE_RE.match(normalized):
            raise ValueError(
                f"Invalid device string: {device_str!r}. "
                "Expected one of: 'auto', 'cpu', 'xpu', 'cuda', 'xpu-<N>', 'cuda-<N>'."
            )
        if "-" in normalized:
            type_str, idx_str = normalized.split("-", 1)
            return DeviceType(type_str), int(idx_str)
        return DeviceType(normalized), None

    def validate(self, device_str: str) -> bool:
        """Return True if the device string is syntactically valid AND currently available."""
        try:
            device_type, index = self.parse(device_str)
        except ValueError:
            return False
        if device_type in (DeviceType.AUTO, DeviceType.CPU):
            return True
        target_index = 0 if index is None else index
        return any(d.type == device_type and d.index == target_index for d in self._devices)

    def resolve(self, device_str: str) -> DeviceInfo:
        """Resolve a stored device preference into a ``DeviceInfo``.

        - ``auto`` returns a synthetic AUTO ``DeviceInfo`` (callers should call
          :meth:`resolve_auto` to obtain a concrete device).
        - Unavailable or unparsable values are logged at WARNING level and fall back to AUTO.
        """
        try:
            device_type, index = self.parse(device_str)
        except ValueError:
            logger.warning("Invalid device string %r; falling back to auto.", device_str)
            return _auto_info()

        if device_type == DeviceType.AUTO:
            return _auto_info()
        if device_type == DeviceType.CPU:
            return _cpu_info()

        target_index = 0 if index is None else index
        match = next((d for d in self._devices if d.type == device_type and d.index == target_index), None)
        if match is None:
            logger.warning("Configured device %r is not available on this system; falling back to auto.", device_str)
            return _auto_info()
        return match

    def resolve_auto(self) -> DeviceInfo:
        """Collapse AUTO to a concrete device.

        Priority: prefer Intel XPU family, then NVIDIA CUDA family. Within a family pick
        the device with the highest total memory; on ties prefer the lowest index. If
        no GPUs are present, fall back to CPU.
        """
        for family in (DeviceType.XPU, DeviceType.CUDA):
            candidates = [d for d in self._devices if d.type == family]
            if candidates:
                return max(candidates, key=lambda d: (d.memory or 0, -(d.index or 0)))
        cpu = next((d for d in self._devices if d.type == DeviceType.CPU), None)
        return cpu or _cpu_info()
