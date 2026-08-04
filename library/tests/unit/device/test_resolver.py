# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for runtime-specific model device resolution."""

import pytest

from instantlearn.device import DeviceInfo, DeviceType, resolve_device_for_runtime
from instantlearn.utils.constants import Backend


def _device(
    device_type: DeviceType,
    name: str,
    index: int | None,
    runtime_ids: dict[Backend, str],
    memory: int | None = None,
) -> DeviceInfo:
    return DeviceInfo(
        type=device_type,
        name=name,
        memory=memory,
        index=index,
        runtime_ids=runtime_ids,
    )


def test_resolve_explicit_device_returns_exact_runtime_id() -> None:
    """An explicit compatible device keeps its authoritative runtime ID."""
    device = _device(
        DeviceType.GPU,
        "Intel Arc",
        0,
        {Backend.TORCH: "xpu:0", Backend.OPENVINO: "GPU.0"},
    )

    selected, runtime_id = resolve_device_for_runtime(
        device,
        Backend.OPENVINO,
        frozenset({DeviceType.GPU}),
    )

    assert selected is device
    assert runtime_id == "GPU.0"


def test_resolve_explicit_device_rejects_unsupported_type() -> None:
    """An explicit device must use a physical type supported by the model."""
    device = _device(DeviceType.NPU, "Intel NPU", 0, {Backend.OPENVINO: "NPU"})

    with pytest.raises(ValueError, match="does not support device type"):
        resolve_device_for_runtime(device, Backend.OPENVINO, frozenset({DeviceType.CPU}))


def test_resolve_explicit_device_rejects_missing_runtime() -> None:
    """An explicit device must be addressable by the requested runtime."""
    device = _device(DeviceType.GPU, "NVIDIA", 0, {Backend.TORCH: "cuda:0"})

    with pytest.raises(ValueError, match="is not available through runtime"):
        resolve_device_for_runtime(device, Backend.OPENVINO, frozenset({DeviceType.GPU}))


def test_resolve_device_rejects_string() -> None:
    """Legacy runtime strings are not part of the strict constructor contract."""
    with pytest.raises(TypeError, match="DeviceInfo instance or None"):
        resolve_device_for_runtime("cpu", Backend.TORCH, frozenset({DeviceType.CPU}))  # type: ignore[arg-type]


def test_resolve_none_selects_best_compatible_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-selection applies type, memory, and index priority deterministically."""
    devices = [
        _device(DeviceType.CPU, "CPU", None, {Backend.TORCH: "cpu"}),
        _device(DeviceType.GPU, "GPU 1", 1, {Backend.TORCH: "cuda:1"}, memory=16_000),
        _device(DeviceType.GPU, "GPU 0", 0, {Backend.TORCH: "cuda:0"}, memory=16_000),
        _device(DeviceType.NPU, "NPU", 0, {Backend.OPENVINO: "NPU"}),
    ]
    monkeypatch.setattr("instantlearn.device.resolver.enumerate_system_devices", lambda: devices)

    selected, runtime_id = resolve_device_for_runtime(
        None,
        Backend.TORCH,
        frozenset({DeviceType.CPU, DeviceType.GPU}),
    )

    assert selected.name == "GPU 0"
    assert runtime_id == "cuda:0"


def test_resolve_none_raises_when_no_compatible_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-selection fails clearly when the runtime has no compatible device."""
    devices = [_device(DeviceType.NPU, "NPU", 0, {Backend.OPENVINO: "NPU"})]
    monkeypatch.setattr("instantlearn.device.resolver.enumerate_system_devices", lambda: devices)

    with pytest.raises(RuntimeError, match="No available device supports runtime 'torch'"):
        resolve_device_for_runtime(None, Backend.TORCH, frozenset({DeviceType.CPU, DeviceType.GPU}))
