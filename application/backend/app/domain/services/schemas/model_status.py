# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Schemas describing the lifecycle state of the inference model (processor).

The status is broadcast from `PipelineManager` to UI subscribers over SSE so users can see when a model
is being (re)loaded after configuration or prompt changes, and so prompt/model controls can be temporarily disabled
while the model is unavailable.
"""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class ModelState(StrEnum):
    """Lifecycle state of the inference model.

    `LOADING_REFERENCE_BATCH` and `LOADING_MODEL` are "busy" states from the UI's perspective
    (controls disabled, banner shown).
    """

    IDLE = "idle"
    LOADING_REFERENCE_BATCH = "loading_reference_batch"
    LOADING_MODEL = "loading_model"
    READY = "ready"
    ERROR = "error"


# Per-state user-facing message templates. Kept centrally so wording can evolve
# without touching transition logic. `LOADING_MODEL` and `READY` accept
# `{model_name}` and `{device}` substitutions.
MODEL_STATUS_MESSAGES: dict[ModelState, str] = {
    ModelState.IDLE: "No active model",
    ModelState.LOADING_REFERENCE_BATCH: "Building reference batch from prompts…",
    ModelState.LOADING_MODEL: "Loading model {model_name} on {device}…",
    ModelState.READY: "Model {model_name} ready on {device}",
}

MODEL_STATUS_ERROR_PREFIX = "Model failed to load"

# Maximum length for sanitized error detail to avoid leaking long internal traces to the UI.
_MAX_ERROR_DETAIL_LEN = 500


def _format(template: str, *, model_name: str | None, device: str | None) -> str:
    """Format a status message, substituting `unknown` for missing values."""
    return template.format(
        model_name=model_name or "unknown",
        device=device or "unknown",
    )


def sanitize_error_detail(detail: str) -> str:
    """Trim and sanitize an exception message for safe UI display."""
    cleaned = detail.strip().replace("\n", " ")
    if len(cleaned) > _MAX_ERROR_DETAIL_LEN:
        cleaned = cleaned[:_MAX_ERROR_DETAIL_LEN].rstrip() + "…"
    return cleaned


class ModelErrorSchema(BaseModel):
    """Structured error info attached to `ModelState.ERROR` snapshots."""

    code: str = Field(description="Exception class name, e.g. 'RuntimeError'.")
    detail: str = Field(description="Human-readable, sanitized error message.")


class ModelStatusSchema(BaseModel):
    """Snapshot of the inference model lifecycle state.

    The same schema is returned by the snapshot endpoint and pushed as the
    `data` payload of each `status` SSE event.
    """

    state: ModelState
    project_id: UUID | None = None
    model_name: str | None = None
    device: str | None = None
    message: str
    error: ModelErrorSchema | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def idle(cls, project_id: UUID | None = None) -> "ModelStatusSchema":
        """Build an IDLE snapshot (no active model)."""
        return cls(
            state=ModelState.IDLE,
            project_id=project_id,
            message=MODEL_STATUS_MESSAGES[ModelState.IDLE],
        )

    @classmethod
    def loading_reference_batch(
        cls,
        *,
        project_id: UUID | None = None,
        model_name: str | None = None,
        device: str | None = None,
    ) -> "ModelStatusSchema":
        """Build a LOADING_REFERENCE_BATCH snapshot."""
        return cls(
            state=ModelState.LOADING_REFERENCE_BATCH,
            project_id=project_id,
            model_name=model_name,
            device=device,
            message=MODEL_STATUS_MESSAGES[ModelState.LOADING_REFERENCE_BATCH],
        )

    @classmethod
    def loading_model(
        cls,
        *,
        project_id: UUID | None = None,
        model_name: str | None = None,
        device: str | None = None,
    ) -> "ModelStatusSchema":
        """Build a LOADING_MODEL snapshot."""
        return cls(
            state=ModelState.LOADING_MODEL,
            project_id=project_id,
            model_name=model_name,
            device=device,
            message=_format(
                MODEL_STATUS_MESSAGES[ModelState.LOADING_MODEL],
                model_name=model_name,
                device=device,
            ),
        )

    @classmethod
    def ready(
        cls,
        *,
        project_id: UUID | None = None,
        model_name: str | None = None,
        device: str | None = None,
    ) -> "ModelStatusSchema":
        """Build a READY snapshot."""
        return cls(
            state=ModelState.READY,
            project_id=project_id,
            model_name=model_name,
            device=device,
            message=_format(
                MODEL_STATUS_MESSAGES[ModelState.READY],
                model_name=model_name,
                device=device,
            ),
        )

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        project_id: UUID | None = None,
        model_name: str | None = None,
        device: str | None = None,
    ) -> "ModelStatusSchema":
        """Build an ERROR snapshot from an exception."""
        detail = sanitize_error_detail(str(exc) or exc.__class__.__name__)
        return cls(
            state=ModelState.ERROR,
            project_id=project_id,
            model_name=model_name,
            device=device,
            message=f"{MODEL_STATUS_ERROR_PREFIX}: {detail}",
            error=ModelErrorSchema(code=type(exc).__name__, detail=detail),
        )
