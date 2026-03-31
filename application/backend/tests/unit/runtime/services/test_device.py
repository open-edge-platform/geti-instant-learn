# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from runtime.services.device import list_available_devices


def test_list_available_devices_enumerates_all_xpu_and_cuda_devices(mocker):
    fake_torch = SimpleNamespace(
        xpu=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            get_device_name=lambda index: f"Intel GPU {index}",
        ),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            get_device_name=lambda index: f"NVIDIA GPU {index}",
        ),
    )
    mocker.patch.dict("sys.modules", {"torch": fake_torch})

    devices = list_available_devices()

    assert [device.device_id for device in devices] == ["xpu:0", "xpu:1", "cuda:0", "cuda:1", "cpu"]
    assert [device.name for device in devices] == [
        "Intel GPU 0",
        "Intel GPU 1",
        "NVIDIA GPU 0",
        "NVIDIA GPU 1",
        "CPU",
    ]
