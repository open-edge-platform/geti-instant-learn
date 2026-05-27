# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from pydantic import BaseModel, Field


class ModelStatus(StrEnum):
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class ModelStatusErrorType(StrEnum):
    AUTH_REQUIRED = "auth_required"
    ACCESS_REQUIRED = "access_required"
    LOAD_FAILED = "load_failed"


class ModelStatusSchema(BaseModel):
    status: ModelStatus = Field(description="Current processor load state.")
    error_type: ModelStatusErrorType | None = Field(
        default=None,
        description="High-level error classification when the last processor load failed.",
    )
    error_message: str | None = Field(
        default=None,
        description="Curated user-facing error message when the last processor load failed.",
    )
