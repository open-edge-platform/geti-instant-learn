# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from api.error_handler import custom_exception_handler
from domain.services.schemas.device import AvailableDeviceSchema
from domain.services.schemas.project import Device


def _create_client(devices: list[AvailableDeviceSchema]) -> TestClient:
    app = FastAPI()
    app.add_exception_handler(Exception, custom_exception_handler)
    app.state.available_devices = devices

    from api.endpoints import devices as _  # noqa: F401
    from api.routers import system_router

    app.include_router(system_router, prefix="/api/v1")
    return TestClient(app, raise_server_exceptions=False)


def test_get_available_devices_cpu_only():
    cpu_device = AvailableDeviceSchema(id=uuid4(), backend=Device.CPU, name="CPU")
    response = _create_client([cpu_device]).get("/api/v1/system/devices")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "devices": [{"id": str(cpu_device.id), "backend": "cpu", "name": "CPU"}],
        "pagination": {"count": 1, "total": 1, "offset": 0, "limit": 20},
    }


def test_get_available_devices_cuda_and_cpu():
    cuda_device = AvailableDeviceSchema(id=uuid4(), backend=Device.CUDA, name="NVIDIA GPU 0")
    cpu_device = AvailableDeviceSchema(id=uuid4(), backend=Device.CPU, name="CPU")
    response = _create_client([cuda_device, cpu_device]).get("/api/v1/system/devices")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "devices": [
            {"id": str(cuda_device.id), "backend": "cuda", "name": "NVIDIA GPU 0"},
            {"id": str(cpu_device.id), "backend": "cpu", "name": "CPU"},
        ],
        "pagination": {"count": 2, "total": 2, "offset": 0, "limit": 20},
    }


def test_get_available_devices_xpu_cuda_and_cpu():
    xpu_device = AvailableDeviceSchema(id=uuid4(), backend=Device.XPU, name="Intel GPU 0")
    cuda_device = AvailableDeviceSchema(id=uuid4(), backend=Device.CUDA, name="NVIDIA GPU 0")
    cpu_device = AvailableDeviceSchema(id=uuid4(), backend=Device.CPU, name="CPU")
    response = _create_client([xpu_device, cuda_device, cpu_device]).get("/api/v1/system/devices")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "devices": [
            {"id": str(xpu_device.id), "backend": "xpu", "name": "Intel GPU 0"},
            {"id": str(cuda_device.id), "backend": "cuda", "name": "NVIDIA GPU 0"},
            {"id": str(cpu_device.id), "backend": "cpu", "name": "CPU"},
        ],
        "pagination": {"count": 3, "total": 3, "offset": 0, "limit": 20},
    }


def test_get_available_devices_with_offset_and_limit():
    xpu_device = AvailableDeviceSchema(id=uuid4(), backend=Device.XPU, name="Intel GPU 0")
    cuda_device = AvailableDeviceSchema(id=uuid4(), backend=Device.CUDA, name="NVIDIA GPU 0")
    cpu_device = AvailableDeviceSchema(id=uuid4(), backend=Device.CPU, name="CPU")
    response = _create_client([xpu_device, cuda_device, cpu_device]).get("/api/v1/system/devices?offset=1&limit=1")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "devices": [{"id": str(cuda_device.id), "backend": "cuda", "name": "NVIDIA GPU 0"}],
        "pagination": {"count": 1, "total": 3, "offset": 1, "limit": 1},
    }
