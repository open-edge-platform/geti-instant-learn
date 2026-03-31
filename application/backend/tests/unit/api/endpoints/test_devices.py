# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
    response = _create_client([AvailableDeviceSchema(backend=Device.CPU, device_id="cpu", name="CPU")]).get(
        "/api/v1/system/devices"
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "devices": [{"backend": "cpu", "device_id": "cpu", "name": "CPU", "index": None}],
        "pagination": {"count": 1, "total": 1, "offset": 0, "limit": 20},
    }


def test_get_available_devices_cuda_and_cpu():
    response = _create_client(
        [
            AvailableDeviceSchema(backend=Device.CUDA, device_id="cuda:0", name="NVIDIA GPU 0", index=0),
            AvailableDeviceSchema(backend=Device.CPU, device_id="cpu", name="CPU"),
        ]
    ).get("/api/v1/system/devices")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "devices": [
            {"backend": "cuda", "device_id": "cuda:0", "name": "NVIDIA GPU 0", "index": 0},
            {"backend": "cpu", "device_id": "cpu", "name": "CPU", "index": None},
        ],
        "pagination": {"count": 2, "total": 2, "offset": 0, "limit": 20},
    }


def test_get_available_devices_xpu_cuda_and_cpu():
    response = _create_client(
        [
            AvailableDeviceSchema(backend=Device.XPU, device_id="xpu:0", name="Intel GPU 0", index=0),
            AvailableDeviceSchema(backend=Device.CUDA, device_id="cuda:0", name="NVIDIA GPU 0", index=0),
            AvailableDeviceSchema(backend=Device.CPU, device_id="cpu", name="CPU"),
        ]
    ).get("/api/v1/system/devices")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "devices": [
            {"backend": "xpu", "device_id": "xpu:0", "name": "Intel GPU 0", "index": 0},
            {"backend": "cuda", "device_id": "cuda:0", "name": "NVIDIA GPU 0", "index": 0},
            {"backend": "cpu", "device_id": "cpu", "name": "CPU", "index": None},
        ],
        "pagination": {"count": 3, "total": 3, "offset": 0, "limit": 20},
    }


def test_get_available_devices_with_offset_and_limit():
    response = _create_client(
        [
            AvailableDeviceSchema(backend=Device.XPU, device_id="xpu:0", name="Intel GPU 0", index=0),
            AvailableDeviceSchema(backend=Device.XPU, device_id="xpu:1", name="Intel GPU 1", index=1),
            AvailableDeviceSchema(backend=Device.CUDA, device_id="cuda:0", name="NVIDIA GPU 0", index=0),
            AvailableDeviceSchema(backend=Device.CPU, device_id="cpu", name="CPU"),
        ]
    ).get("/api/v1/system/devices?offset=1&limit=2")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "devices": [
            {"backend": "xpu", "device_id": "xpu:1", "name": "Intel GPU 1", "index": 1},
            {"backend": "cuda", "device_id": "cuda:0", "name": "NVIDIA GPU 0", "index": 0},
        ],
        "pagination": {"count": 2, "total": 4, "offset": 1, "limit": 2},
    }
