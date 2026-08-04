# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for cross-runtime device discovery."""

import sys
from types import SimpleNamespace
from typing import ClassVar

import pytest

from instantlearn.device.device import (
    DeviceType,
    discover_openvino_devices,
    discover_torch_devices,
    merge_device_observations,
)
from instantlearn.utils.constants import Backend


def test_torch_discovery_does_not_advertise_cpu_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Torch CPU is absent when the optional Torch package is unavailable."""
    monkeypatch.setitem(sys.modules, "torch", None)

    assert discover_torch_devices() == []


def test_discovery_merges_same_gpu_by_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime observations with the same UUID form one physical GPU."""
    fake_torch = SimpleNamespace(
        xpu=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda _index: "Intel Arc",
            get_device_properties=lambda _index: SimpleNamespace(total_memory=16_000, uuid="abc-123"),
        ),
        cuda=SimpleNamespace(is_available=lambda: False),
    )

    class FakeCore:
        available_devices: ClassVar[list[str]] = ["CPU", "GPU.0", "NPU"]

        def get_property(self, device_id: str, property_name: str) -> str | None:
            values = {
                ("CPU", "FULL_DEVICE_NAME"): "CPU",
                ("GPU.0", "FULL_DEVICE_NAME"): "Intel Arc",
                ("GPU.0", "DEVICE_UUID"): "abc123",
                ("NPU", "FULL_DEVICE_NAME"): "Intel NPU",
                ("NPU", "DEVICE_UUID"): "npu123",
            }
            return values.get((device_id, property_name))

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=FakeCore))

    devices = merge_device_observations([*discover_torch_devices(), *discover_openvino_devices()])

    assert [device.type for device in devices] == [DeviceType.GPU, DeviceType.NPU, DeviceType.CPU]
    assert devices[0].runtime_ids == {Backend.TORCH: "xpu:0", Backend.OPENVINO: "GPU.0"}
    assert devices[1].runtime_ids == {Backend.OPENVINO: "NPU"}
    assert devices[2].runtime_ids == {Backend.TORCH: "cpu", Backend.OPENVINO: "CPU"}


def test_discovery_does_not_merge_different_gpus_with_same_runtime_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal runtime indices alone do not identify the same physical GPU."""
    fake_torch = SimpleNamespace(
        xpu=SimpleNamespace(is_available=lambda: False),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda _index: "NVIDIA RTX",
            get_device_properties=lambda _index: SimpleNamespace(total_memory=24_000, uuid=None),
        ),
    )

    class FakeCore:
        available_devices: ClassVar[list[str]] = ["GPU.0"]

        def get_property(self, device_id: str, property_name: str) -> str | None:
            return "Intel Graphics" if (device_id, property_name) == ("GPU.0", "FULL_DEVICE_NAME") else None

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=FakeCore))

    devices = merge_device_observations([*discover_torch_devices(), *discover_openvino_devices()])

    assert len([device for device in devices if device.type == DeviceType.GPU]) == 2
