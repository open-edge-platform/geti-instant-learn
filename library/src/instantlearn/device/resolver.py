# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Model-aware runtime and device selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from instantlearn.utils.constants import Backend

from .device import DeviceInfo, DeviceType, get_supported_devices

if TYPE_CHECKING:
    from collections.abc import Sequence

    from instantlearn.models.model_card import ModelCard

_TYPE_PRIORITY = {
    DeviceType.CPU: 0,
    DeviceType.NPU: 1,
    DeviceType.GPU: 2,
}


@dataclass(frozen=True)
class ResolvedDevice:
    """Concrete runtime route selected for a model."""

    device: DeviceInfo
    runtime: Backend
    runtime_id: str
    fallback_used: bool = False


def _device_priority(device: DeviceInfo) -> tuple[int, int, int]:
    return (
        _TYPE_PRIORITY[device.type],
        device.memory or 0,
        -(device.index or 0),
    )


def _resolve_route(
    model_card: ModelCard,
    device: DeviceInfo,
    allowed_runtimes: tuple[Backend, ...],
    *,
    fallback_used: bool = False,
) -> ResolvedDevice | None:
    for runtime in allowed_runtimes:
        runtime_id = device.runtime_id(runtime)
        if runtime_id is not None and model_card.supports(runtime, device.type):
            return ResolvedDevice(
                device=device,
                runtime=runtime,
                runtime_id=runtime_id,
                fallback_used=fallback_used,
            )
    return None


def resolve_device_for_model(
    model_card: ModelCard,
    device: DeviceInfo | None,
    *,
    devices: Sequence[DeviceInfo] | None = None,
    allowed_runtimes: tuple[Backend, ...] = (Backend.OPENVINO, Backend.TORCH),
    allow_fallback: bool = False,
) -> ResolvedDevice:
    """Resolve a physical device and runtime supported by a model.

    Args:
        model_card: Capabilities used to validate runtime and device support.
        device: Preferred physical device, or ``None`` to select automatically.
        devices: Available devices. Discovers the local system when omitted.
        allowed_runtimes: Runtime priority, from highest to lowest.
        allow_fallback: Select another device when the preferred one is incompatible.

    Returns:
        The selected physical device, runtime, and exact runtime identifier.

    Raises:
        TypeError: If ``device`` is neither ``DeviceInfo`` nor ``None``.
        ValueError: If an explicit device is incompatible and fallback is disabled.
        RuntimeError: If no available device can run the model.
    """
    if device is not None and not isinstance(device, DeviceInfo):
        msg = "device must be a DeviceInfo instance or None"
        raise TypeError(msg)

    if device is not None:
        route = _resolve_route(model_card, device, allowed_runtimes)
        if route is not None:
            return route
        if not allow_fallback:
            runtime_names = ", ".join(runtime.value for runtime in allowed_runtimes)
            msg = f"Device {device.key!r} cannot run model {model_card.name!r} through: {runtime_names}."
            raise ValueError(msg)

    available_devices = list(devices) if devices is not None else get_supported_devices()
    candidates = sorted(
        (candidate for candidate in available_devices if candidate != device),
        key=_device_priority,
        reverse=True,
    )
    for runtime in allowed_runtimes:
        for candidate in candidates:
            route = _resolve_route(
                model_card,
                candidate,
                (runtime,),
                fallback_used=device is not None,
            )
            if route is not None:
                return route

    msg = f"No available device can run model {model_card.name!r}."
    raise RuntimeError(msg)
