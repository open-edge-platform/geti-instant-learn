# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from api.error_handler import custom_exception_handler
from domain.services.schemas.device import DeviceInfo, DeviceType
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
    cpu_device = DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)
    response = _create_client([cpu_device]).get("/api/v1/system/devices")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == [{"type": "cpu", "name": "CPU", "memory": None, "index": None}]


def test_get_available_devices_cuda_and_cpu():
    cuda_device = DeviceInfo(type=DeviceType.CUDA, name="NVIDIA GPU 0", memory=25_000_000_000, index=0)
    cpu_device = DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)
    response = _create_client([cuda_device, cpu_device]).get("/api/v1/system/devices")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == [
        {"type": "cuda", "name": "NVIDIA GPU 0", "memory": 25_000_000_000, "index": 0},
        {"type": "cpu", "name": "CPU", "memory": None, "index": None},
    ]


def test_get_available_devices_xpu_cuda_and_cpu():
    xpu_device = DeviceInfo(type=DeviceType.XPU, name="Intel GPU 0", memory=16_000_000_000, index=0)
    cuda_device = DeviceInfo(type=DeviceType.CUDA, name="NVIDIA GPU 0", memory=25_000_000_000, index=0)
    cpu_device = DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)
    response = _create_client([xpu_device, cuda_device, cpu_device]).get("/api/v1/system/devices")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == [
        {"type": "xpu", "name": "Intel GPU 0", "memory": 16_000_000_000, "index": 0},
        {"type": "cuda", "name": "NVIDIA GPU 0", "memory": 25_000_000_000, "index": 0},
        {"type": "cpu", "name": "CPU", "memory": None, "index": None},
    ]
