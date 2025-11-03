# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import re

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from core.runtime.errors import PipelineNotActiveError, PipelineProjectMismatchError
from exceptions.custom_errors import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceUpdateConflictError,
)

logger = logging.getLogger(__name__)


def custom_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Centralized exception handler for FastAPI routes.
    Maps domain exceptions to appropriate HTTP status codes and returns consistent error responses.

    Args:
        request: The incoming request object.
        exc: The exception object.

    Returns:
        JSONResponse with appropriate status code and error message.
    """
    try:
        body_str = request._body.decode("utf-8") if hasattr(request, "_body") and request._body else ""
    except Exception:
        body_str = "<unable to read body>"

    if isinstance(exc, ResourceNotFoundError):
        logger.debug(
            f"Exception handler called: {request.method} {request.url.path} "
            f"raised {type(exc).__name__}: {str(exc)}. Body: {body_str}"
        )
        message = str(exc) if str(exc) else "The requested resource was not found."
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": message})

    if isinstance(exc, ResourceAlreadyExistsError):
        logger.debug(
            f"Exception handler called: {request.method} {request.url.path} "
            f"raised {type(exc).__name__}: {str(exc)}. Body: {body_str}"
        )
        message = str(exc) if str(exc) else "A conflict occurred with the current state of the resource."
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": message})

    if isinstance(
        exc, (ResourceUpdateConflictError | PipelineNotActiveError | PipelineProjectMismatchError | ValueError)
    ):
        logger.debug(
            f"Exception handler called: {request.method} {request.url.path} "
            f"raised {type(exc).__name__}: {str(exc)}. Body: {body_str}"
        )
        message = str(exc) if str(exc) else "Invalid request. Please check your input and try again."
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": message})

    if isinstance(exc, RequestValidationError):
        return _handle_validation_error(request, exc, body_str)

    if isinstance(exc, IntegrityError):
        logger.error(f"Unhandled IntegrityError in endpoint: {exc}", exc_info=exc)
        message = "Database constraint violation. Please check your input."
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": message})

    logger.error(
        f"Internal error for {request.method} {request.url.path}: "
        f"{type(exc).__name__}: {str(exc)}. "
        f"Headers: {dict(request.headers)}. Body: {body_str}",
        exc_info=exc,
    )
    message = "An internal server error occurred. Please try again later or contact support for assistance."
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": message})


def _handle_validation_error(request: Request, exc: RequestValidationError, body_str: str) -> JSONResponse:
    """
    Handle Pydantic validation errors with user-friendly messages.
    Returns 400 instead of 422 for better client handling.

    Args:
        request: The incoming request object.
        exc: The RequestValidationError exception.
        body_str: Pre-read request body string.

    Returns:
        JSONResponse with formatted validation errors.
    """
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


def extract_constraint_name(error_msg: str) -> str | None:
    """
    Extract constraint name from SQLAlchemy IntegrityError message.

    Args:
        error_msg: The error message from exc.orig

    Returns:
        Constraint name if found, else None
    """
    error_msg = error_msg.lower()

    # try direct constraint name match
    for pattern in [
        r"constraint failed:\s*(\w+)",
        r"constraint\s*['\"](\w+)['\"]",
        r"constraint\s+(\w+)\s+failed",
    ]:
        match = re.search(pattern, error_msg)
        if match:
            return match.group(1)

    # try table.column format for implicit constraints
    match = re.search(r"(\w+)\.(\w+)", error_msg)
    if match:
        return f"{match.group(1)}_{match.group(2)}"

    return None
