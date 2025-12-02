# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class BaseIDPayload(BaseModel):
    """Base payload with an id field generated, if not provided."""

    id: UUID = Field(default_factory=uuid4)


class BaseIDSchema(BaseModel):
    """Base model with an id field."""

    id: UUID


class Pagination(BaseModel):
    """Pagination model."""

    count: int  # number of items actually returned (might be less than limit if at the end)
    total: int  # total number of items available
    offset: int = 0  # index of the first item returned (0-based)
    limit: int = 10  # number of items requested per page


class PaginatedResponse(BaseModel):
    """Base paginated response model."""

    pagination: Pagination
