# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import status

from api.routers import license_router
from domain.services.schemas.license import LicenseAcceptedSchema
from instantlearn_license.service import LicenseService

logger = logging.getLogger(__name__)


@license_router.post(
    path="/accept",
    tags=["License"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "License accepted successfully or was already accepted.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Failed to persist license acceptance.",
        },
    },
)
def accept_license() -> LicenseAcceptedSchema:
    """Accept the third-party license terms"""
    service = LicenseService()

    if not service.is_accepted():
        service.accept()
        logger.info("License accepted via API endpoint")
    else:
        logger.debug("License was already accepted")

    return LicenseAcceptedSchema(accepted=True)
