# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from domain.services.schemas.device import DeviceInfo, DeviceType
from runtime.services.device import DeviceService, enumerate_system_devices


def _make_device(type_: DeviceType, name: str, memory: int | None = None, index: int | None = None) -> DeviceInfo:
    return DeviceInfo(type=type_, name=name, memory=memory, index=index)


def test_enumerate_system_devices_lists_xpu_cuda_and_cpu(mocker):
    fake_torch = SimpleNamespace(
        xpu=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            get_device_name=lambda index: f"Intel GPU {index}",
            get_device_properties=lambda index: SimpleNamespace(total_memory=1_000 * (index + 1)),
        ),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda index: f"NVIDIA GPU {index}",
            get_device_properties=lambda index: SimpleNamespace(total_memory=25_000_000_000),
        ),
    )
    mocker.patch.dict("sys.modules", {"torch": fake_torch})

    devices = enumerate_system_devices()

    assert [d.type for d in devices] == [DeviceType.XPU, DeviceType.XPU, DeviceType.CUDA, DeviceType.CPU]
    assert [d.name for d in devices] == ["Intel GPU 0", "Intel GPU 1", "NVIDIA GPU 0", "CPU"]
    assert [d.index for d in devices] == [0, 1, 0, None]
    assert devices[0].memory == 1_000
    assert devices[1].memory == 2_000
    assert devices[2].memory == 25_000_000_000
    assert devices[3].memory is None


@pytest.mark.parametrize(
    ("device_str", "expected_type", "expected_index"),
    [
        ("auto", DeviceType.AUTO, None),
        ("cpu", DeviceType.CPU, None),
        ("xpu", DeviceType.XPU, None),
        ("cuda", DeviceType.CUDA, None),
        ("xpu-0", DeviceType.XPU, 0),
        ("cuda-3", DeviceType.CUDA, 3),
        ("CUDA-1", DeviceType.CUDA, 1),
    ],
)
def test_parse_valid_strings(device_str, expected_type, expected_index):
    assert DeviceService.parse(device_str) == (expected_type, expected_index)


@pytest.mark.parametrize(
    "device_str",
    ["", "auto-0", "cpu-1", "gpu", "xpu-", "xpu-abc", "cuda:0", "tpu"],
)
def test_parse_invalid_strings_raise(device_str):
    with pytest.raises(ValueError):
        DeviceService.parse(device_str)


def test_validate_known_and_unknown_devices():
    service = DeviceService(
        devices=[
            _make_device(DeviceType.XPU, "Intel", memory=8_000, index=0),
            _make_device(DeviceType.CPU, "CPU"),
        ]
    )
    assert service.validate("auto") is True
    assert service.validate("cpu") is True
    assert service.validate("xpu") is True
    assert service.validate("xpu-0") is True
    assert service.validate("xpu-1") is False
    assert service.validate("cuda") is False
    assert service.validate("garbage") is False


def test_resolve_returns_auto_for_unknown(caplog):
    service = DeviceService(devices=[_make_device(DeviceType.CPU, "CPU")])
    with caplog.at_level("WARNING"):
        info = service.resolve("cuda-2")
    assert info.type == DeviceType.AUTO
    assert "not available" in caplog.text


def test_resolve_returns_concrete_for_known():
    xpu = _make_device(DeviceType.XPU, "Intel", memory=8_000, index=0)
    service = DeviceService(devices=[xpu, _make_device(DeviceType.CPU, "CPU")])
    assert service.resolve("xpu-0") == xpu
    assert service.resolve("xpu") == xpu  # default to index 0
    assert service.resolve("cpu").type == DeviceType.CPU
    assert service.resolve("auto").type == DeviceType.AUTO


def test_resolve_auto_prefers_xpu_then_cuda_then_cpu():
    xpu = _make_device(DeviceType.XPU, "Intel A", memory=8_000, index=0)
    cuda_big = _make_device(DeviceType.CUDA, "NVIDIA 4090", memory=25_000_000_000, index=0)
    cpu = _make_device(DeviceType.CPU, "CPU")

    assert DeviceService(devices=[xpu, cuda_big, cpu]).resolve_auto() == xpu
    assert DeviceService(devices=[cuda_big, cpu]).resolve_auto() == cuda_big
    assert DeviceService(devices=[cpu]).resolve_auto() == cpu


def test_resolve_auto_picks_highest_memory_then_lowest_index_within_family():
    xpu_low = _make_device(DeviceType.XPU, "Intel A", memory=8_000_000_000, index=1)
    xpu_high = _make_device(DeviceType.XPU, "Intel B", memory=16_000_000_000, index=2)
    cpu = _make_device(DeviceType.CPU, "CPU")
    assert DeviceService(devices=[xpu_low, xpu_high, cpu]).resolve_auto() == xpu_high

    xpu_idx0 = _make_device(DeviceType.XPU, "Intel A", memory=8_000_000_000, index=0)
    xpu_idx1 = _make_device(DeviceType.XPU, "Intel B", memory=8_000_000_000, index=1)
    assert DeviceService(devices=[xpu_idx1, xpu_idx0, cpu]).resolve_auto() == xpu_idx0


def test_resolve_auto_falls_back_to_cpu_when_no_gpus():
    cpu = _make_device(DeviceType.CPU, "CPU")
    assert DeviceService(devices=[cpu]).resolve_auto() == cpu


def test_resolve_auto_with_empty_devices_returns_synthetic_cpu():
    info = DeviceService(devices=[]).resolve_auto()
    assert info.type == DeviceType.CPU
    assert info.name == "CPU"
