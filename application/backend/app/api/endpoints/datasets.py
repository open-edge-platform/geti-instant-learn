# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import status

from api.routers import system_router
from dependencies import AvailableDatasetsDep
from domain.services.schemas.dataset import DatasetsListSchema


@system_router.get(
    path="/datasets",
    tags=["System"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successfully retrieved available datasets."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Unexpected error occurred."},
    },
)
def get_datasets(available_datasets: AvailableDatasetsDep) -> DatasetsListSchema:
    """
    List datasets metadata available for download.
    Return startup-static dataset metadata cache (no runtime filesystem rescan).
    """
    return available_datasets
