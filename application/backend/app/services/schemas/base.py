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
