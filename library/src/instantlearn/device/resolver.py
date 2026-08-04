# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime-specific model device selection."""

from instantlearn.utils.constants import Backend

from .device import DeviceInfo, DeviceType, enumerate_system_devices

_TYPE_PRIORITY = {
    DeviceType.CPU: 0,
    DeviceType.NPU: 1,
    DeviceType.GPU: 2,
}


def _validate_device(
    device: DeviceInfo,
    runtime: Backend,
    supported_device_types: frozenset[DeviceType],
) -> tuple[DeviceInfo, str]:
    if device.type not in supported_device_types:
        msg = f"Runtime {runtime.value!r} does not support device type {device.type.value!r} for this model."
        raise ValueError(msg)

    runtime_id = device.runtime_id(runtime)
    if runtime_id is None:
        msg = f"Device {device.key!r} is not available through runtime {runtime.value!r}."
        raise ValueError(msg)
    return device, runtime_id


def _device_priority(device: DeviceInfo) -> tuple[int, int, int]:
    return (
        _TYPE_PRIORITY[device.type],
        device.memory or 0,
        -(device.index or 0),
    )


def resolve_device_for_runtime(
    device: DeviceInfo | None,
    runtime: Backend,
    supported_device_types: frozenset[DeviceType],
) -> tuple[DeviceInfo, str]:
    """Resolve an explicit or automatic model device for one runtime.

    Args:
        device: Explicit physical device, or ``None`` to select automatically.
        runtime: Runtime used by the model implementation.
        supported_device_types: Physical device types accepted by the model.

    Returns:
        The selected physical device and its exact runtime identifier.

    Raises:
        TypeError: If ``device`` is neither ``DeviceInfo`` nor ``None``.
        RuntimeError: If automatic selection finds no compatible device.
    """
    if device is not None and not isinstance(device, DeviceInfo):
        msg = "device must be a DeviceInfo instance or None"
        raise TypeError(msg)
    if device is not None:
        return _validate_device(device, runtime, supported_device_types)

    candidates = [
        candidate
        for candidate in enumerate_system_devices()
        if candidate.type in supported_device_types and candidate.runtime_id(runtime) is not None
    ]
    if not candidates:
        msg = f"No available device supports runtime {runtime.value!r} for this model."
        raise RuntimeError(msg)

    selected = max(candidates, key=_device_priority)
    return _validate_device(selected, runtime, supported_device_types)
