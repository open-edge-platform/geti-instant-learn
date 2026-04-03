# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from fastapi import Query, status

from api.routers import system_router
from dependencies import AvailableDevicesDep
from domain.services.schemas.base import Pagination
from domain.services.schemas.device import DevicesListSchema


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
    available_devices: AvailableDevicesDep,
    offset: Annotated[int, Query(ge=0, le=1000)] = 0,
    limit: Annotated[int, Query(ge=0, le=1000)] = 20,
) -> DevicesListSchema:
    """List available runtime devices (e.g. CUDA, XPU, CPU)."""
    total = len(available_devices)
    paged_devices = available_devices[offset : offset + limit]
    return DevicesListSchema(
        devices=paged_devices,
        pagination=Pagination(count=len(paged_devices), total=total, offset=offset, limit=limit),
    )
