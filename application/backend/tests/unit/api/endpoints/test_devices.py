# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from instantlearn.device import DeviceInfo, DeviceType
from instantlearn.utils.constants import Backend

from api.error_handler import custom_exception_handler
from runtime.services.device import DeviceService


def _create_client(devices: list[DeviceInfo]) -> TestClient:
    app = FastAPI()
    app.add_exception_handler(Exception, custom_exception_handler)
    app.state.device_service = DeviceService(devices=devices)

    from api.endpoints import devices as _  # noqa: F401
    from api.routers import system_router

    app.include_router(system_router, prefix="/api/v1")
    return TestClient(app, raise_server_exceptions=False)


def test_get_available_devices_cpu_only():
    cpu_device = DeviceInfo(
        type=DeviceType.CPU,
        name="CPU",
        memory=None,
        index=None,
        runtime_ids={Backend.TORCH: "cpu", Backend.OPENVINO: "CPU"},
    )
    response = _create_client([cpu_device]).get("/api/v1/system/devices")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == [
        {
            "type": "cpu",
            "name": "CPU",
            "memory": None,
            "index": None,
            "key": "cpu",
            "runtime_ids": {"torch": "cpu", "openvino": "CPU"},
        }
    ]


def test_get_available_devices_exposes_runtime_ids():
    gpu_device = DeviceInfo(
        type=DeviceType.GPU,
        name="Intel GPU 0",
        memory=16_000_000_000,
        index=0,
        runtime_ids={Backend.TORCH: "xpu:0", Backend.OPENVINO: "GPU.0"},
    )
    npu_device = DeviceInfo(
        type=DeviceType.NPU,
        name="Intel NPU",
        memory=None,
        index=0,
        runtime_ids={Backend.OPENVINO: "NPU"},
    )
    response = _create_client([gpu_device, npu_device]).get("/api/v1/system/devices")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == [
        {
            "type": "gpu",
            "name": "Intel GPU 0",
            "memory": 16_000_000_000,
            "index": 0,
            "key": "gpu-0",
            "runtime_ids": {"torch": "xpu:0", "openvino": "GPU.0"},
        },
        {
            "type": "npu",
            "name": "Intel NPU",
            "memory": None,
            "index": 0,
            "key": "npu-0",
            "runtime_ids": {"openvino": "NPU"},
        },
    ]
