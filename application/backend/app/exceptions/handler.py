# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from uuid import UUID

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from db.constraints import CheckConstraintName, UniqueConstraintName
from exceptions.custom_errors import (
    PipelineNotActiveError,
    PipelineProjectMismatchError,
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
    ResourceUpdateConflictError,
)

logger = logging.getLogger(__name__)


async def custom_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Centralized exception handler for FastAPI routes.
    Maps domain exceptions to appropriate HTTP status codes and returns consistent error responses.

    Args:
        request: The incoming request object.
        exc: The exception object.

    Returns:
        JSONResponse with appropriate status code and error message.
    """
    if isinstance(exc, ResourceNotFoundError):
        return await _handle_exception(request, exc, status.HTTP_404_NOT_FOUND)

    if isinstance(exc, ResourceAlreadyExistsError):
        return await _handle_exception(request, exc, status.HTTP_409_CONFLICT)

    if isinstance(
        exc, (ResourceUpdateConflictError | PipelineNotActiveError | PipelineProjectMismatchError | ValueError)
    ):
        return await _handle_exception(request, exc, status.HTTP_400_BAD_REQUEST)

    if isinstance(exc, RequestValidationError):
        return await _handle_validation_error(request, exc)

    if isinstance(exc, IntegrityError):
        logger.error(f"Unhandled IntegrityError in endpoint: {exc}", exc_info=exc)
        return await _handle_exception(
            request,
            ValueError("Database constraint violation. Please check your input."),
            status.HTTP_400_BAD_REQUEST,
        )

    return await _handle_exception(request, exc, status.HTTP_500_INTERNAL_SERVER_ERROR)


async def _handle_exception(request: Request, exc: Exception, status_code: int) -> JSONResponse:
    """
    Handle general exceptions with appropriate logging and response formatting.

    Args:
        request: The incoming request object.
        exc: The exception object.
        status_code: HTTP status code to return.

    Returns:
        JSONResponse with error details.
    """
    try:
        body = await request.body()
        body_str = body.decode("utf-8") if body else ""
    except Exception:
        body_str = "<unable to read body>"

    logger.debug(
        f"Exception handler called: {request.method} {request.url.path} "
        f"raised {type(exc).__name__}: {str(exc)}. "
        f"Body: {body_str}"
    )

    match status_code:
        case status.HTTP_400_BAD_REQUEST:
            message = str(exc) if str(exc) else "Invalid request. Please check your input and try again."
        case status.HTTP_404_NOT_FOUND:
            message = str(exc) if str(exc) else "The requested resource was not found."
        case status.HTTP_409_CONFLICT:
            message = str(exc) if str(exc) else "A conflict occurred with the current state of the resource."
        case _:
            logger.error(
                f"Internal error for {request.method} {request.url.path}: "
                f"{type(exc).__name__}: {str(exc)}. "
                f"Headers: {dict(request.headers)}. Body: {body_str}",
                exc_info=exc,
            )
            message = "An internal server error occurred. Please try again later or contact support for assistance."

    return JSONResponse(status_code=status_code, content={"detail": message})


async def _handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle Pydantic validation errors with user-friendly messages.
    Returns 400 instead of 422 for better client handling.

    Args:
        request: The incoming request object.
        exc: The RequestValidationError exception.

    Returns:
        JSONResponse with formatted validation errors.
    """
    try:
        body = await request.body()
        body_str = body.decode("utf-8") if body else ""
    except Exception:
        body_str = "<unable to read body>"

    logger.debug(f"Validation error for {request.method} {request.url.path}: {exc.errors()}. Body: {body_str}")

    error_messages = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        error_type = error["type"]
        msg = error["msg"]

        if error_type == "missing":
            error_messages.append(f"Field '{field_path}' is required.")
        elif error_type in ("string_type", "int_type", "float_type", "bool_type"):
            error_messages.append(f"Field '{field_path}' has invalid type: {msg}")
        else:
            error_messages.append(f"Field '{field_path}': {msg}")

    detail = " ".join(error_messages) if error_messages else "Invalid request data."

    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": detail})


def handle_integrity_error(exc: IntegrityError, resource_type: ResourceType, resource_id: UUID | None = None) -> None:
    """
    Map SQLAlchemy IntegrityError to appropriate custom exception.
    This function should be called in service layer try-except blocks.

    Args:
        exc: The IntegrityError exception
        resource_type: Type of resource being operated on
        resource_id: ID of the resource (if available)

    Raises:
        ResourceAlreadyExistsError: For UNIQUE constraint violations
        ResourceNotFoundError: For FOREIGN KEY constraint violations
        ValueError: For CHECK or NOT NULL constraint violations
    """
    error_msg = str(exc.orig).lower()  # the original database exception object

    if "unique constraint failed" in error_msg:
        _handle_unique_constraint(error_msg, resource_type)
    elif "foreign key constraint failed" in error_msg:
        raise ResourceNotFoundError(
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id else None,
            message="Referenced resource does not exist.",
        )
    elif "check constraint failed" in error_msg:
        _handle_check_constraint(error_msg)
    elif "not null constraint failed" in error_msg:
        _handle_not_null_constraint(error_msg)
    else:
        logger.error(f"Unmapped IntegrityError: {error_msg}", exc_info=exc)
        raise ValueError(f"Database constraint violation for {resource_type.value}.")


def _handle_unique_constraint(error_msg: str, resource_type: ResourceType) -> None:
    """Handle UNIQUE constraint violations."""
    constraint_messages = {
        UniqueConstraintName.PROJECT_NAME: ("name", "A project with this name already exists."),
        UniqueConstraintName.PROMPT_NAME_PER_PROJECT: (
            "name",
            "A prompt with this name already exists in the project.",
        ),
        UniqueConstraintName.PROCESSOR_NAME_PER_PROJECT: (
            "name",
            "A processor with this name already exists in the project.",
        ),
        UniqueConstraintName.SOURCE_NAME_PER_PROJECT: (
            "name",
            "A source with this name already exists in the project.",
        ),
        UniqueConstraintName.SOURCE_TYPE_PER_PROJECT: (
            "source_type",
            "A source with this type already exists in the project.",
        ),
        UniqueConstraintName.LABEL_NAME_PER_PROJECT: ("name", "A label with this name already exists in the project."),
        UniqueConstraintName.SINGLE_ACTIVE_PROJECT: ("active", "Only one project can be active at a time."),
        UniqueConstraintName.SINGLE_CONNECTED_SOURCE_PER_PROJECT: (
            "connected",
            "Only one source can be connected per project at a time.",
        ),
    }

    for constraint, (field_name, message) in constraint_messages.items():
        if constraint in error_msg or constraint.value.replace("uq_", "").replace("_", "") in error_msg:
            raise ResourceAlreadyExistsError(
                resource_type=resource_type,
                resource_value=field_name,
                raised_by="name",
                message=message,
            )

    logger.warning(f"Unmapped unique constraint violation: {error_msg}")
    raise ResourceAlreadyExistsError(
        resource_type=resource_type,
        resource_value="unknown",
        raised_by="name",
        message=f"{resource_type.value} already exists.",
    )


def _handle_check_constraint(error_msg: str) -> None:
    """Handle CHECK constraint violations using constraint name enums."""
    if CheckConstraintName.LABEL_PARENT in error_msg:
        raise ValueError("Label must belong to either a project or a prompt.")

    constraint_match = re.search(r"check constraint failed:\s*(\w+)", error_msg)
    constraint_name = constraint_match.group(1) if constraint_match else "unknown"
    logger.warning(f"Unmapped check constraint violation: {constraint_name}")
    raise ValueError(f"Validation failed: {constraint_name}")


def _handle_not_null_constraint(error_msg: str) -> None:
    """Handle NOT NULL constraint violations."""
    column_match = re.search(r"not null constraint failed:\s*\w+\.(\w+)", error_msg)
    column_name = column_match.group(1) if column_match else "unknown field"
    raise ValueError(f"Required field '{column_name}' cannot be null.")
