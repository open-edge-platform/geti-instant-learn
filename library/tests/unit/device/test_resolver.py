# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for model-aware runtime and device resolution."""

import pytest

from instantlearn.device import DeviceInfo, DeviceType, resolve_device_for_model
from instantlearn.models.model_card import ModelCard, RuntimeCapability
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


def _card(*capabilities: RuntimeCapability) -> ModelCard:
    return ModelCard(
        name="Test model",
        family="test",
        description="test",
        prompt_types=frozenset(),
        shot_modes=frozenset(),
        exportable_to=frozenset(),
        supported_runtimes=frozenset(capabilities),
    )


def test_resolve_explicit_device_returns_exact_runtime_id() -> None:
    """An explicit compatible device keeps its authoritative runtime ID."""
    device = _device(
        DeviceType.GPU,
        "Intel Arc",
        0,
        {Backend.TORCH: "xpu:0", Backend.OPENVINO: "GPU.0"},
    )

    card = _card(RuntimeCapability(Backend.OPENVINO, frozenset({DeviceType.GPU})))

    resolved = resolve_device_for_model(
        card,
        device,
        allowed_runtimes=(Backend.OPENVINO,),
    )

    assert resolved.device is device
    assert resolved.runtime == Backend.OPENVINO
    assert resolved.runtime_id == "GPU.0"
    assert resolved.fallback_used is False


def test_resolve_explicit_device_rejects_unsupported_type() -> None:
    """An explicit device must use a physical type supported by the model."""
    device = _device(DeviceType.NPU, "Intel NPU", 0, {Backend.OPENVINO: "NPU"})
    card = _card(RuntimeCapability(Backend.OPENVINO, frozenset({DeviceType.CPU})))

    with pytest.raises(ValueError, match="cannot run model"):
        resolve_device_for_model(card, device, allowed_runtimes=(Backend.OPENVINO,))


def test_resolve_explicit_device_rejects_missing_runtime() -> None:
    """An explicit device must be addressable by the requested runtime."""
    device = _device(DeviceType.GPU, "NVIDIA", 0, {Backend.TORCH: "cuda:0"})
    card = _card(RuntimeCapability(Backend.OPENVINO, frozenset({DeviceType.GPU})))

    with pytest.raises(ValueError, match="cannot run model"):
        resolve_device_for_model(card, device, allowed_runtimes=(Backend.OPENVINO,))


def test_resolve_device_rejects_string() -> None:
    """Legacy runtime strings are not part of the strict constructor contract."""
    card = _card(RuntimeCapability(Backend.TORCH, frozenset({DeviceType.CPU})))

    with pytest.raises(TypeError, match="DeviceInfo instance or None"):
        resolve_device_for_model(card, "cpu")  # type: ignore[arg-type]


def test_resolve_none_selects_best_compatible_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-selection applies type, memory, and index priority deterministically."""
    devices = [
        _device(DeviceType.CPU, "CPU", None, {Backend.TORCH: "cpu"}),
        _device(DeviceType.GPU, "GPU 1", 1, {Backend.TORCH: "cuda:1"}, memory=16_000),
        _device(DeviceType.GPU, "GPU 0", 0, {Backend.TORCH: "cuda:0"}, memory=16_000),
        _device(DeviceType.NPU, "NPU", 0, {Backend.OPENVINO: "NPU"}),
    ]
    monkeypatch.setattr("instantlearn.device.resolver.get_supported_devices", lambda: devices)
    card = _card(RuntimeCapability(Backend.TORCH, frozenset({DeviceType.CPU, DeviceType.GPU})))

    resolved = resolve_device_for_model(
        card,
        None,
        allowed_runtimes=(Backend.TORCH,),
    )

    assert resolved.device.name == "GPU 0"
    assert resolved.runtime_id == "cuda:0"


def test_resolve_none_raises_when_no_compatible_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-selection fails clearly when the runtime has no compatible device."""
    devices = [_device(DeviceType.NPU, "NPU", 0, {Backend.OPENVINO: "NPU"})]
    monkeypatch.setattr("instantlearn.device.resolver.get_supported_devices", lambda: devices)
    card = _card(RuntimeCapability(Backend.TORCH, frozenset({DeviceType.CPU, DeviceType.GPU})))

    with pytest.raises(RuntimeError, match="No available device can run model"):
        resolve_device_for_model(card, None, allowed_runtimes=(Backend.TORCH,))


def test_resolve_prefers_runtime_order_before_device_priority() -> None:
    """Runtime priority is applied before physical device priority."""
    devices = [
        _device(DeviceType.CPU, "CPU", None, {Backend.OPENVINO: "CPU"}),
        _device(DeviceType.GPU, "GPU", 0, {Backend.TORCH: "cuda:0"}),
    ]
    card = _card(
        RuntimeCapability(Backend.OPENVINO, frozenset({DeviceType.CPU})),
        RuntimeCapability(Backend.TORCH, frozenset({DeviceType.GPU})),
    )

    resolved = resolve_device_for_model(card, None, devices=devices)

    assert resolved.device.type == DeviceType.CPU
    assert resolved.runtime == Backend.OPENVINO


def test_resolve_falls_back_from_incompatible_preference() -> None:
    """Fallback mode selects another compatible route and reports the fallback."""
    preferred = _device(DeviceType.NPU, "NPU", 0, {Backend.OPENVINO: "NPU"})
    cpu = _device(DeviceType.CPU, "CPU", None, {Backend.TORCH: "cpu"})
    card = _card(RuntimeCapability(Backend.TORCH, frozenset({DeviceType.CPU})))

    resolved = resolve_device_for_model(
        card,
        preferred,
        devices=[preferred, cpu],
        allowed_runtimes=(Backend.TORCH,),
        allow_fallback=True,
    )

    assert resolved.device is cpu
    assert resolved.runtime_id == "cpu"
    assert resolved.fallback_used is True
