# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import status

from api.routers import source_types_router
from dependencies import DiscoveryServiceDep
from domain.services.schemas.reader import ReaderConfig

logger = logging.getLogger(__name__)


@source_types_router.get(
    path="/{source_type}/sources",
    tags=["Source Types"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successfully retrieved available sources."},
        status.HTTP_400_BAD_REQUEST: {"description": "Discovery not supported for this source type."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Unexpected error occurred."},
    },
)
def get_available_sources(source_type: str, discovery_service: DiscoveryServiceDep) -> list[ReaderConfig]:
    """
    List available sources for the specified type (e.g. connected usb cameras).
    """
    return discovery_service.list_available_sources(source_type)
