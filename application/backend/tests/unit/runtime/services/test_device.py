# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from instantlearn.device import DeviceInfo, DeviceType
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
        ("GPU-1", ("gpu", 1)),
        ("npu", ("npu", None)),
    ],
)
def test_parse_valid_device_keys(device_str, expected):
    assert DeviceService.parse(device_str) == expected


@pytest.mark.parametrize(
    "device_str",
    ["", "auto-0", "cpu-1", "gpu-", "npu:0", "tpu", "xpu", "xpu-0", "cuda", "cuda-0"],
)
def test_parse_invalid_device_keys(device_str):
    with pytest.raises(ValueError):
        DeviceService.parse(device_str)


def test_validate_accepts_only_available_physical_keys(devices):
    service = DeviceService(devices)

    assert service.validate("auto") is True
    assert service.validate("cpu") is True
    assert service.validate("gpu-0") is True
    assert service.validate("npu-0") is True
    assert service.validate("gpu-3") is False


def test_resolve_preference_returns_selected_physical_device(devices):
    resolved = DeviceService(devices).resolve_preference("gpu-0")

    assert resolved is devices[0]


def test_resolve_preference_returns_none_for_auto(devices):
    assert DeviceService(devices).resolve_preference("auto") is None


@pytest.mark.parametrize("device_str", ["invalid", "gpu-9"])
def test_resolve_preference_returns_none_for_invalid_or_unavailable_device(devices, device_str):
    assert DeviceService(devices).resolve_preference(device_str) is None
