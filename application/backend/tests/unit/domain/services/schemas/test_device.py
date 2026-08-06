# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from instantlearn.device import DeviceInfo as LibraryDeviceInfo
from instantlearn.device import DeviceType
from instantlearn.utils.constants import Backend

from domain.services.schemas.device import DeviceInfo


def test_from_library_preserves_physical_device_and_runtime_ids() -> None:
    library_device = LibraryDeviceInfo(
        type=DeviceType.GPU,
        name="Intel Graphics",
        memory=8_000_000_000,
        index=1,
        runtime_ids={Backend.TORCH: "xpu:1", Backend.OPENVINO: "GPU.1"},
    )

    device = DeviceInfo.from_library(library_device)

    assert device.type == DeviceType.GPU
    assert device.name == "Intel Graphics"
    assert device.memory == 8_000_000_000
    assert device.index == 1
    assert device.key == "gpu-1"
    assert device.runtime_ids == {Backend.TORCH: "xpu:1", Backend.OPENVINO: "GPU.1"}


def test_model_dump_serializes_runtime_ids_for_api() -> None:
    device = DeviceInfo.from_library(
        LibraryDeviceInfo(
            type=DeviceType.CPU,
            name="CPU",
            runtime_ids={Backend.TORCH: "cpu", Backend.OPENVINO: "CPU"},
        ),
    )

    assert device.model_dump(mode="json") == {
        "type": "cpu",
        "name": "CPU",
        "memory": None,
        "index": None,
        "key": "cpu",
        "runtime_ids": {"torch": "cpu", "openvino": "CPU"},
    }
