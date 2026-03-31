# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from api.error_handler import custom_exception_handler


def _create_client(devices: list[str]) -> TestClient:
    app = FastAPI()
    app.add_exception_handler(Exception, custom_exception_handler)
    app.state.available_devices = devices

    from api.endpoints import devices as _  # noqa: F401
    from api.routers import system_router

    app.include_router(system_router, prefix="/api/v1")
    return TestClient(app, raise_server_exceptions=False)


def test_get_available_devices_cpu_only():
    response = _create_client(["cpu"]).get("/api/v1/system/devices")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "devices": ["cpu"],
        "pagination": {"count": 1, "total": 1, "offset": 0, "limit": 20},
    }


def test_get_available_devices_cuda_and_cpu():
    response = _create_client(["cuda", "cpu"]).get("/api/v1/system/devices")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "devices": ["cuda", "cpu"],
        "pagination": {"count": 2, "total": 2, "offset": 0, "limit": 20},
    }


def test_get_available_devices_xpu_cuda_and_cpu():
    response = _create_client(["xpu", "cuda", "cpu"]).get("/api/v1/system/devices")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "devices": ["xpu", "cuda", "cpu"],
        "pagination": {"count": 3, "total": 3, "offset": 0, "limit": 20},
    }


def test_get_available_devices_with_offset_and_limit():
    response = _create_client(["xpu", "cuda", "cpu"]).get("/api/v1/system/devices?offset=1&limit=1")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "devices": ["cuda"],
        "pagination": {"count": 1, "total": 3, "offset": 1, "limit": 1},
    }
