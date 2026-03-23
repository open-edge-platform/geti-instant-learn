# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from domain.services.schemas.base import BaseIDSchema, PaginatedResponse, Pagination


class DatasetSchema(BaseIDSchema):
    """Public dataset metadata returned by API."""

    name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=500)


class DatasetsListSchema(PaginatedResponse):
    """Wrapper schema for datasets list responses."""

    datasets: list[DatasetSchema]


def empty_datasets_list() -> DatasetsListSchema:
    """Create an empty datasets list with consistent pagination metadata."""
    return DatasetsListSchema(datasets=[], pagination=Pagination(count=0, total=0, offset=0, limit=0))
