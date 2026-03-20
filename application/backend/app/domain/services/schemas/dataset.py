# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from pydantic import BaseModel, Field


class DatasetSchema(BaseModel):
    """Public dataset metadata returned by API."""

    id: UUID
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=500)


class DatasetsListSchema(BaseModel):
    """Wrapper schema for datasets list responses."""

    datasets: list[DatasetSchema]
