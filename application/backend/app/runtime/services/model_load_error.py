# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from domain.services.schemas.model_status import ModelStatusErrorType

_HF_AUTH_ERROR_MESSAGE = (
    "Model loading failed because Hugging Face authentication is required. Run `hf auth login`, then try again."
)
_HF_ACCESS_ERROR_MESSAGE = (
    "Model loading failed because access to this Hugging Face model has not been granted. "
    "Request access for the model on Hugging Face and try again."
)
_GENERIC_MODEL_LOAD_ERROR_MESSAGE = "Model loading failed. Check the backend logs for details and try again."
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
    "please log in",
)


def _exception_chain_contains(exc: Exception, markers: tuple[str, ...]) -> bool:
    """Return True when any exception in the cause/context chain contains a marker.

    Libraries such as `transformers` and `huggingface_hub` often wrap the
    original access/auth failure in a higher-level exception. Walking the
    chain keeps classification stable when the top-level exception message is
    generic but the nested cause still contains the useful Hugging Face
    wording.
    """
    pending: list[BaseException] = [exc]
    visited: set[int] = set()

    while pending:
        current = pending.pop()
        current_id = id(current)
        if current_id in visited:
            continue
        visited.add(current_id)

        message = str(current).lower()
        if any(marker in message for marker in markers):
            return True

        cause = getattr(current, "__cause__", None)
        if cause is not None:
            pending.append(cause)

        context = getattr(current, "__context__", None)
        if context is not None:
            pending.append(context)

    return False


def is_huggingface_access_error(exc: Exception) -> bool:
    """Return True when the exception indicates gated-model access has not been granted."""
    return _exception_chain_contains(exc, _HF_ACCESS_ERROR_MARKERS)


def is_huggingface_auth_error(exc: Exception) -> bool:
    """Return True when the exception indicates Hugging Face authentication is missing."""
    return _exception_chain_contains(exc, _HF_AUTH_ERROR_MARKERS)


def build_model_load_error(exc: Exception) -> tuple[ModelStatusErrorType, str]:
    """Classify a model-load failure and return the corresponding user-facing status payload."""
    if is_huggingface_access_error(exc):
        return ModelStatusErrorType.ACCESS_REQUIRED, _HF_ACCESS_ERROR_MESSAGE
    if is_huggingface_auth_error(exc):
        return ModelStatusErrorType.AUTH_REQUIRED, _HF_AUTH_ERROR_MESSAGE
    return ModelStatusErrorType.LOAD_FAILED, _GENERIC_MODEL_LOAD_ERROR_MESSAGE
