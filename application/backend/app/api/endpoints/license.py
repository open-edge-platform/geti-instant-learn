# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import status

from api.routers import system_router
from dependencies import LicenseServiceDep
from domain.services.schemas.license import LicenseAcceptedSchema

logger = logging.getLogger(__name__)


@system_router.post(
    path="/license/accept",
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
def accept_license(license_service: LicenseServiceDep) -> LicenseAcceptedSchema:
    """Accept the third-party license terms"""
    if not license_service.is_accepted():
        license_service.accept()
        logger.info("Geti Instant Learn License has been accepted via API endpoint")
    else:
        logger.debug("License was already accepted")

    return LicenseAcceptedSchema(accepted=True)
