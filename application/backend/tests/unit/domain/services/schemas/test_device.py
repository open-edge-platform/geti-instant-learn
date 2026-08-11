# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from instantlearn.device import DeviceInfo, DeviceType
from instantlearn.utils.constants import Backend


def test_device_info_preserves_physical_device_and_runtime_ids() -> None:
    device = DeviceInfo(
        type=DeviceType.GPU,
        name="Intel Graphics",
        memory=8_000_000_000,
        index=1,
        runtime_ids={Backend.TORCH: "xpu:1", Backend.OPENVINO: "GPU.1"},
    )

    assert device.type == DeviceType.GPU
    assert device.name == "Intel Graphics"
    assert device.memory == 8_000_000_000
    assert device.index == 1
    assert device.key == "gpu-1"
    assert device.runtime_ids == {Backend.TORCH: "xpu:1", Backend.OPENVINO: "GPU.1"}


def test_model_dump_serializes_runtime_ids_for_api() -> None:
    device = DeviceInfo(
        type=DeviceType.CPU,
        name="CPU",
        runtime_ids={Backend.TORCH: "cpu", Backend.OPENVINO: "CPU"},
    )

    assert device.model_dump(mode="json") == {
        "type": "cpu",
        "name": "CPU",
        "memory": None,
        "index": None,
        "key": "cpu",
        "runtime_ids": {"torch": "cpu", "openvino": "CPU"},
    }


def test_model_dump_excludes_internal_identity() -> None:
    device = DeviceInfo(
        type=DeviceType.GPU,
        name="GPU",
        index=0,
        runtime_ids={Backend.OPENVINO: "GPU.0"},
        identity="internal-uuid",
    )

    assert "identity" not in device.model_dump(mode="json")
