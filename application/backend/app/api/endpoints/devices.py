# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import status

from api.routers import system_router
from dependencies import DeviceServiceDep
from domain.services.schemas.device import DeviceInfo


@system_router.get(
    path="/devices",
    tags=["System"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successfully retrieved available devices."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Unexpected error occurred."},
    },
)
def get_available_devices(
    device_service: DeviceServiceDep,
) -> list[DeviceInfo]:
    """List available runtime devices (e.g. CUDA, XPU, CPU)."""
    return device_service.list_devices()
