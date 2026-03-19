# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import status

from api.routers import system_router
from dependencies import DatasetRegistryServiceDep
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
def get_datasets(dataset_registry_service: DatasetRegistryServiceDep) -> DatasetsListSchema:
    """List datasets metadata available for download."""
    return dataset_registry_service.list_datasets()
