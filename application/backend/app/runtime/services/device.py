# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from dataclasses import dataclass

from instantlearn.device import DeviceInfo, DeviceType, enumerate_system_devices
from instantlearn.models.model_card import ModelCard
from instantlearn.utils.constants import Backend

from domain.services.schemas.device import DEVICE_STR_PATTERN

logger = logging.getLogger(__name__)

_DEVICE_RE = re.compile(DEVICE_STR_PATTERN)


@dataclass(frozen=True)
class ResolvedDevice:
    """Concrete runtime route selected for a model."""

    device: DeviceInfo
    runtime: Backend
    runtime_id: str
    fallback_used: bool = False


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
    def parse(device_str: str) -> tuple[str, int | None]:
        """Parse a device selection key into ``(kind, index)``.

        Raises:
            ValueError: When the string doesn't match the canonical format.
        """
        normalized = device_str.lower()
        if not _DEVICE_RE.match(normalized):
            raise ValueError(
                f"Invalid device string: {device_str!r}. Expected 'auto', 'cpu', "
                "'<gpu|npu|xpu|cuda>', or '<gpu|npu|xpu|cuda>-<N>'."
            )
        if "-" in normalized:
            type_str, idx_str = normalized.split("-", 1)
            return type_str, int(idx_str)
        return normalized, None

    def _find_device(self, kind: str, index: int | None) -> DeviceInfo | None:
        if kind == DeviceType.CPU.value:
            return next((device for device in self._devices if device.type == DeviceType.CPU), None)

        target_index = 0 if index is None else index
        if kind in (DeviceType.GPU.value, DeviceType.NPU.value):
            device_type = DeviceType(kind)
            return next(
                (device for device in self._devices if device.type == device_type and device.index == target_index),
                None,
            )

        return next(
            (device for device in self._devices if device.runtime_ids.get(Backend.TORCH) == f"{kind}:{target_index}"),
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

    def resolve(self, device_str: str) -> DeviceInfo:
        """Resolve a stored device preference into a ``DeviceInfo``.

        Unavailable, unparsable, and ``auto`` values resolve to the best concrete device.
        """
        try:
            kind, index = self.parse(device_str)
        except ValueError:
            logger.warning("Invalid device string %r; falling back to auto.", device_str)
            return self.resolve_auto()

        if kind == "auto":
            return self.resolve_auto()

        match = self._find_device(kind, index)
        if match is None:
            logger.warning("Configured device %r is not available on this system; falling back to auto.", device_str)
            return self.resolve_auto()
        return match

    def resolve_auto(self) -> DeviceInfo:
        """Collapse AUTO to a concrete device.

        Prefer GPU, then NPU, then CPU. Within a class prefer devices with more
        memory, then the lowest registry index.
        """
        for device_type in (DeviceType.GPU, DeviceType.NPU, DeviceType.CPU):
            candidates = [device for device in self._devices if device.type == device_type]
            if candidates:
                return max(candidates, key=lambda device: (device.memory or 0, -(device.index or 0)))
        raise RuntimeError("No runtime devices were discovered.")

    def resolve_for_model(
        self,
        model_card: ModelCard,
        device_str: str = "auto",
        allowed_runtimes: tuple[Backend, ...] = (Backend.OPENVINO, Backend.TORCH),
    ) -> ResolvedDevice:
        """Resolve a device preference to a runtime supported by ``model_card``."""
        invalid_preference = False
        try:
            kind, index = self.parse(device_str)
        except ValueError:
            kind, index = "auto", None
            invalid_preference = True

        preferred = None if kind == "auto" else self._find_device(kind, index)
        if preferred is not None:
            for runtime in allowed_runtimes:
                runtime_id = preferred.runtime_id(runtime)
                if runtime_id is not None and model_card.supports(runtime, preferred.type):
                    return ResolvedDevice(
                        device=preferred,
                        runtime=runtime,
                        runtime_id=runtime_id,
                    )

        fallback_used = invalid_preference or kind != "auto"
        candidates = [
            device for device in self._ordered_devices() if preferred is None or device != preferred
        ]
        for runtime in allowed_runtimes:
            for device in candidates:
                runtime_id = device.runtime_id(runtime)
                if runtime_id is None or not model_card.supports(runtime, device.type):
                    continue
                if fallback_used:
                    logger.warning(
                        "Device %r is not supported by model %s; using %s on %s.",
                        device_str,
                        model_card.name,
                        runtime.value,
                        device.key,
                    )
                return ResolvedDevice(
                    device=device,
                    runtime=runtime,
                    runtime_id=runtime_id,
                    fallback_used=fallback_used,
                )

        raise RuntimeError(f"No available device can run model {model_card.name!r}.")

    def _ordered_devices(self) -> list[DeviceInfo]:
        type_priority = {DeviceType.GPU: 0, DeviceType.NPU: 1, DeviceType.CPU: 2}
        return sorted(
            self._devices,
            key=lambda device: (
                type_priority[device.type],
                -(device.memory or 0),
                device.index or 0,
            ),
        )
