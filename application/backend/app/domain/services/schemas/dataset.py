# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from domain.services.schemas.base import BaseIDSchema, PaginatedResponse


class DatasetSchema(BaseIDSchema):
    """Public dataset metadata returned by API."""

    name: str = Field(min_length=1, max_length=120)
    thumbnail: str | None = None


class DatasetsListSchema(PaginatedResponse):
    """Wrapper schema for datasets list responses."""

    datasets: list[DatasetSchema]
