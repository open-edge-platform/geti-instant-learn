# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from exceptions.custom_errors import (
    PipelineNotActiveError,
    PipelineProjectMismatchError,
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceUpdateConflictError, ResourceType,
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
    # Map specific exceptions to status codes
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

    # unknown exceptions
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
        body = await request.body()  # read request body for logging (if available)
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


def handle_integrity_error(exc: IntegrityError, resource_type: ResourceType, resource_id: UUID | None) -> None:
    """
    Map SQLAlchemy IntegrityError to appropriate custom exception.

    Args:
        exc: The IntegrityError exception
        resource_type: Type of resource being operated on
        resource_id: ID of the resource (if available)

    Raises:
        ResourceAlreadyExistsError: For UNIQUE constraint violations
        ResourceNotFoundError: For FOREIGN KEY constraint violations
        ValueError: For CHECK or NOT NULL constraint violations
    """
    error_msg = str(exc.orig).lower()

    if "unique constraint failed" in error_msg:
        field_match = error_msg.split(":")[-1].strip() if ":" in error_msg else None  # todo that's not needed here maybe??
        raise ResourceAlreadyExistsError(
            resource_type=resource_type,
            resource_value=str(resource_id) if resource_id else "unknown",
            raised_by="id",
            message=f"{resource_type.value} with ID {resource_id} already exists."
        )

    if "foreign key constraint failed" in error_msg:
        raise ResourceNotFoundError(
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id else None,
            message=f"Referenced resource does not exist."
        )

    if "check constraint failed" in error_msg or "not null constraint failed" in error_msg:
        raise ValueError(f"Invalid data: {error_msg}")

    logger.error(f"Unmapped integrity error: {exc}", exc_info=exc)
    raise