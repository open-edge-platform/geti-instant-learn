# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from instantlearn.device import DeviceInfo, DeviceType
from instantlearn.models.model_card import ModelCard, RuntimeCapability
from instantlearn.utils.constants import Backend

from runtime.services.device import DeviceService


def _device(
    type_: DeviceType,
    name: str,
    index: int | None,
    runtime_ids: dict[Backend, str],
    memory: int | None = None,
) -> DeviceInfo:
    return DeviceInfo(type=type_, name=name, memory=memory, index=index, runtime_ids=runtime_ids)


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


@pytest.fixture
def devices() -> list[DeviceInfo]:
    return [
        _device(
            DeviceType.GPU,
            "Intel Arc",
            0,
            {Backend.TORCH: "xpu:0", Backend.OPENVINO: "GPU.0"},
            memory=16_000,
        ),
        _device(DeviceType.GPU, "NVIDIA GPU", 1, {Backend.TORCH: "cuda:0"}, memory=24_000),
        _device(DeviceType.NPU, "Intel NPU", 0, {Backend.OPENVINO: "NPU"}),
        _device(DeviceType.CPU, "CPU", None, {Backend.TORCH: "cpu", Backend.OPENVINO: "CPU"}),
    ]


@pytest.mark.parametrize(
    ("device_str", "expected"),
    [
        ("auto", ("auto", None)),
        ("cpu", ("cpu", None)),
        ("gpu-0", ("gpu", 0)),
        ("npu", ("npu", None)),
        ("xpu-0", ("xpu", 0)),
        ("CUDA-0", ("cuda", 0)),
    ],
)
def test_parse_valid_device_keys(device_str, expected):
    assert DeviceService.parse(device_str) == expected


@pytest.mark.parametrize("device_str", ["", "auto-0", "cpu-1", "gpu-", "npu:0", "tpu"])
def test_parse_invalid_device_keys(device_str):
    with pytest.raises(ValueError):
        DeviceService.parse(device_str)


def test_validate_accepts_physical_keys_and_legacy_runtime_aliases(devices):
    service = DeviceService(devices)

    assert service.validate("auto") is True
    assert service.validate("cpu") is True
    assert service.validate("gpu-0") is True
    assert service.validate("npu-0") is True
    assert service.validate("xpu-0") is True
    assert service.validate("cuda-0") is True
    assert service.validate("gpu-3") is False


def test_resolve_for_model_prefers_openvino_on_selected_device(devices):
    card = _card(
        RuntimeCapability(Backend.TORCH, frozenset({DeviceType.CPU, DeviceType.GPU})),
        RuntimeCapability(Backend.OPENVINO, frozenset({DeviceType.CPU, DeviceType.GPU, DeviceType.NPU})),
    )

    resolved = DeviceService(devices).resolve_for_model(card, "xpu-0")

    assert resolved.device.name == "Intel Arc"
    assert resolved.runtime == Backend.OPENVINO
    assert resolved.runtime_id == "GPU.0"
    assert resolved.fallback_used is False


def test_resolve_for_model_auto_applies_runtime_priority_before_memory(devices):
    card = _card(
        RuntimeCapability(Backend.TORCH, frozenset({DeviceType.GPU})),
        RuntimeCapability(Backend.OPENVINO, frozenset({DeviceType.GPU})),
    )

    resolved = DeviceService(devices).resolve_for_model(card, "auto")

    assert resolved.device.name == "Intel Arc"
    assert resolved.runtime == Backend.OPENVINO
    assert resolved.runtime_id == "GPU.0"


def test_resolve_for_model_uses_npu_for_openvino_only_model(devices):
    card = _card(RuntimeCapability(Backend.OPENVINO, frozenset({DeviceType.NPU})))

    resolved = DeviceService(devices).resolve_for_model(card, "auto")

    assert resolved.device.type == DeviceType.NPU
    assert resolved.runtime == Backend.OPENVINO
    assert resolved.runtime_id == "NPU"


def test_resolve_for_model_falls_back_when_preference_is_unsupported(devices):
    card = _card(RuntimeCapability(Backend.TORCH, frozenset({DeviceType.CPU})))

    resolved = DeviceService(devices).resolve_for_model(card, "npu-0")

    assert resolved.device.type == DeviceType.CPU
    assert resolved.runtime_id == "cpu"
    assert resolved.fallback_used is True


def test_resolve_for_model_falls_back_when_device_lacks_required_runtime(devices):
    card = _card(RuntimeCapability(Backend.OPENVINO, frozenset({DeviceType.GPU})))

    resolved = DeviceService(devices).resolve_for_model(card, "gpu-1")

    assert resolved.device.name == "Intel Arc"
    assert resolved.runtime == Backend.OPENVINO
    assert resolved.fallback_used is True


def test_resolve_for_model_marks_invalid_preference_as_fallback(devices):
    card = _card(RuntimeCapability(Backend.TORCH, frozenset({DeviceType.CPU})))

    resolved = DeviceService(devices).resolve_for_model(card, "invalid")

    assert resolved.device.type == DeviceType.CPU
    assert resolved.fallback_used is True


def test_resolve_for_model_respects_allowed_runtimes(devices):
    card = _card(
        RuntimeCapability(Backend.TORCH, frozenset({DeviceType.GPU})),
        RuntimeCapability(Backend.OPENVINO, frozenset({DeviceType.GPU})),
    )

    resolved = DeviceService(devices).resolve_for_model(
        card,
        "gpu-0",
        allowed_runtimes=(Backend.TORCH,),
    )

    assert resolved.runtime == Backend.TORCH
    assert resolved.runtime_id == "xpu:0"


def test_resolve_for_model_raises_when_no_compatible_route(devices):
    card = _card(RuntimeCapability(Backend.OPENVINO, frozenset({DeviceType.NPU})))
    torch_only_devices = [device for device in devices if Backend.OPENVINO not in device.runtime_ids]

    with pytest.raises(RuntimeError, match="No available device"):
        DeviceService(torch_only_devices).resolve_for_model(card)
