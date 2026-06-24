# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Iterator

from domain.services.schemas.model_status import ModelStatusErrorType

_GENERIC_MODEL_LOAD_ERROR_MESSAGE = "Model loading failed. Check the backend logs for details and try again."
_TRACEBACK_EXCEPTION_HEADER_PATTERN = re.compile(
    r"^(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception|Warning|Interrupt|Exit|Iteration):\s*(.*)$"
)
_HF_ACCESS_ERROR_MARKERS = (
    "ask for access",
    "not in the authorized list",
    "requires approved access",
    "does not have access to the weights",
    "request access on the huggingface website",
    "must have access to it and be authenticated",
)
_HF_AUTH_ERROR_MARKERS = (
    "cannot access gated repo",
    "you are trying to access a gated repo",
    "fine-grained token settings",
    "please log in",
    "public gated repositories",
    "token has the correct permissions",
)
_HF_ACTIONABLE_ERROR_MARKERS = _HF_ACCESS_ERROR_MARKERS + _HF_AUTH_ERROR_MARKERS


def _iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    pending: list[BaseException] = [exc]
    visited: set[int] = set()

    while pending:
        current = pending.pop()
        current_id = id(current)
        if current_id in visited:
            continue
        visited.add(current_id)

        yield current

        cause = getattr(current, "__cause__", None)
        if cause is not None:
            pending.append(cause)

        context = getattr(current, "__context__", None)
        if context is not None:
            pending.append(context)


def _exception_chain_contains(exc: Exception, markers: tuple[str, ...]) -> bool:
    """Return True when any exception in the cause/context chain contains a marker.

    Libraries such as `transformers` and `huggingface_hub` often wrap the
    original access/auth failure in a higher-level exception. Walking the
    chain keeps classification stable when the top-level exception message is
    generic but the nested cause still contains the useful Hugging Face
    wording.
    """
    for current in _iter_exception_chain(exc):
        message = str(current).lower()
        if any(marker in message for marker in markers):
            return True

    return False


def _contains_marker(message: str, markers: tuple[str, ...]) -> bool:
    normalized_message = message.lower()
    return any(marker in normalized_message for marker in markers)


def _traceback_line_content(line: str) -> str:
    line = line.removeprefix("[Server] ")
    return line.removeprefix("ERROR: ")


def _traceback_exception_messages(message: str) -> list[str]:
    if "Traceback (most recent call last):" not in message:
        return []

    lines = [_traceback_line_content(line.rstrip()) for line in message.splitlines()]
    exception_messages: list[str] = []

    for index, line in enumerate(lines):
        match = _TRACEBACK_EXCEPTION_HEADER_PATTERN.match(line)
        if match is None:
            continue

        message_lines = [match.group(1).strip()]
        for following_line in lines[index + 1 :]:
            if following_line.startswith(("The above exception was", "During handling of")):
                break
            message_lines.append(following_line)

        exception_message = "\n".join(message_lines).strip()
        if exception_message:
            exception_messages.append(exception_message)

    return exception_messages


def _traceback_error_message(message: str) -> str | None:
    exception_messages = _traceback_exception_messages(message)
    if not exception_messages:
        return None

    for exception_message in reversed(exception_messages):
        if _contains_marker(exception_message, _HF_ACTIONABLE_ERROR_MARKERS):
            return exception_message

    return exception_messages[-1]


def _model_load_error_message(exc: Exception) -> str:
    messages: list[str] = []

    for current in _iter_exception_chain(exc):
        message = str(current).strip()
        if message:
            messages.append(_traceback_error_message(message) or message)

    for message in messages:
        if _contains_marker(message, _HF_ACTIONABLE_ERROR_MARKERS):
            return message

    if messages:
        return messages[0]

    return _GENERIC_MODEL_LOAD_ERROR_MESSAGE


def is_huggingface_access_error(exc: Exception) -> bool:
    """Return True when the exception indicates gated-model access has not been granted."""
    return _exception_chain_contains(exc, _HF_ACCESS_ERROR_MARKERS)


def is_huggingface_auth_error(exc: Exception) -> bool:
    """Return True when the exception indicates Hugging Face authentication is missing."""
    return _exception_chain_contains(exc, _HF_AUTH_ERROR_MARKERS)


def model_load_error(exc: Exception) -> tuple[ModelStatusErrorType, str]:
    """Classify a model-load failure and return the corresponding user-facing status payload."""
    error_message = _model_load_error_message(exc)
    if is_huggingface_access_error(exc):
        return ModelStatusErrorType.ACCESS_REQUIRED, error_message
    if is_huggingface_auth_error(exc):
        return ModelStatusErrorType.AUTH_REQUIRED, error_message
    return ModelStatusErrorType.LOAD_FAILED, error_message
