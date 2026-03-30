# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from fastapi import Query, status

from api.routers import system_router
from dependencies import AvailableDatasetsDep
from domain.services.schemas.base import Pagination
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
def get_datasets(
    available_datasets: AvailableDatasetsDep,
    offset: Annotated[int, Query(ge=0, le=1000)] = 0,
    limit: Annotated[int, Query(ge=0, le=1000)] = 20,
) -> DatasetsListSchema:
    """
    List datasets metadata available for download.
    Return startup-static dataset metadata cache (no runtime filesystem rescan).
    """
    total = len(available_datasets.datasets)
    paged_datasets = available_datasets.datasets[offset : offset + limit]
    return DatasetsListSchema(
        datasets=paged_datasets,
        pagination=Pagination(count=len(paged_datasets), total=total, offset=offset, limit=limit),
    )
