# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from domain.services.schemas.model_status import ModelStatusErrorType
from runtime.services.model_load_error import model_load_error


def traceback_message(*exception_blocks: tuple[str, str]) -> str:
    return "\n\nThe above exception was the direct cause of the following exception:\n\n".join(
        f"Traceback (most recent call last):\n"
        f'  File "<test traceback>", line 1, in test_failure\n'
        f"    raise_error()\n"
        f"{exception_name}: {message}"
        for exception_name, message in exception_blocks
    )


DINO_V3_ACCESS_ERROR_MESSAGE = (
    "User does not have access to the weights of the DinoV3 model.\n"
    "Please follow these steps:\n"
    "1. Request access on the HuggingFace website: "
    "https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m\n"
    "2. Set your HuggingFace credentials using one of these methods:\n"
    "   - Run: hf auth login\n"
    "   - Set environment variable: export HUGGINGFACE_HUB_TOKEN=your_token"
)
SAM3_GATED_REPO_ERROR_MESSAGE = (
    "You are trying to access a gated repo.\n"
    "Make sure to have access to it at https://huggingface.co/facebook/sam3.1.\n"
    "401 Client Error. (Request ID: Root=1-6a071118-35ebe7bb0ae57d6643a0928a;"
    "0052c2c2-7481-4ff6-87fa-fc75b533a121)\n"
    "\n"
    "Cannot access gated repo for url "
    "https://huggingface.co/facebook/sam3.1/resolve/main/tokenizer_config.json.\n"
    "Access to model facebook/sam3.1 is restricted. You must have access to it and be authenticated to access it. "
    "Please log in."
)
SAM3_NOT_AUTHORIZED_ERROR_MESSAGE = (
    "You are trying to access a gated repo.\n"
    "Make sure to have access to it at https://huggingface.co/facebook/sam3.1.\n"
    "403 Client Error. (Request ID: Root=1-6a0afa96-6dce46b641adee9152464dbf;"
    "722a9be1-cb2a-4d6a-a68d-3577569e55aa)\n"
    "\n"
    "Cannot access gated repo for url "
    "https://huggingface.co/facebook/sam3.1/resolve/main/tokenizer_config.json.\n"
    "Access to model facebook/sam3.1 is restricted and you are not in the authorized list. "
    "Visit https://huggingface.co/facebook/sam3.1 to ask for access."
)
SAM3_TOKEN_PERMISSION_ERROR_MESSAGE = (
    "(Request ID: Root=1-6a3ba7ff-732dd29b6497480074539f89;422c4b2e-aad5-43d9-9fcd-3a526924a596)\n"
    "\n"
    "403 Forbidden: Please enable access to public gated repositories in your fine-grained token settings to view "
    "this repository..\n"
    "Cannot access content at: https://huggingface.co/facebook/sam3.1/resolve/main/sam3.1_multiplex.pt.\n"
    "Make sure your token has the correct permissions."
)
LOCAL_CACHE_WRAPPER_MESSAGE = (
    "An error happened while trying to locate the file on the Hub and we cannot find the requested files in the "
    "local cache. Please check your connection and try again or make sure your Internet connection is on."
)


def test_wrapped_huggingface_auth_failure_is_classified_as_auth_required():
    exc = OSError(
        "You are trying to access a gated repo. Make sure to have access to it at "
        "https://huggingface.co/facebook/sam3.1. Please log in."
    )

    error_type, error_message = model_load_error(exc)

    assert error_type == ModelStatusErrorType.AUTH_REQUIRED
    assert error_message == str(exc)


def test_wrapped_huggingface_access_failure_is_classified_as_access_required():
    exc = OSError(
        "Cannot access gated repo for url https://huggingface.co/facebook/sam3.1/resolve/main/tokenizer_config.json. "
        "Access to model facebook/sam3.1 is restricted and you are not in the authorized list. "
        "Visit https://huggingface.co/facebook/sam3.1 to ask for access."
    )

    error_type, error_message = model_load_error(exc)

    assert error_type == ModelStatusErrorType.ACCESS_REQUIRED
    assert error_message == str(exc)


def test_huggingface_value_error_access_failure_is_classified_as_access_required():
    exc = ValueError(DINO_V3_ACCESS_ERROR_MESSAGE)

    error_type, error_message = model_load_error(exc)

    assert error_type == ModelStatusErrorType.ACCESS_REQUIRED
    assert error_message == DINO_V3_ACCESS_ERROR_MESSAGE
    assert "User does not have access to the weights of the DinoV3 model." in error_message
    assert "https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m" in error_message
    assert "hf auth login" in error_message


def test_traceback_message_extracts_terminal_exception_message():
    exc = RuntimeError(
        traceback_message(
            (
                "requests.exceptions.HTTPError",
                "401 Client Error: Unauthorized for url: "
                "https://huggingface.co/facebook/sam3.1/resolve/main/tokenizer_config.json",
            ),
            ("OSError", SAM3_GATED_REPO_ERROR_MESSAGE),
        )
    )

    error_type, error_message = model_load_error(exc)

    assert error_type == ModelStatusErrorType.ACCESS_REQUIRED
    assert error_message == SAM3_GATED_REPO_ERROR_MESSAGE
    assert "Traceback (most recent call last):" not in error_message
    assert not error_message.startswith("OSError:")


def test_traceback_message_extracts_terminal_not_authorized_exception_message():
    exc = RuntimeError(
        traceback_message(
            (
                "requests.exceptions.HTTPError",
                "403 Client Error: Forbidden for url: "
                "https://huggingface.co/facebook/sam3.1/resolve/main/tokenizer_config.json",
            ),
            ("huggingface_hub.errors.GatedRepoError", SAM3_NOT_AUTHORIZED_ERROR_MESSAGE),
            ("OSError", SAM3_NOT_AUTHORIZED_ERROR_MESSAGE),
        )
    )

    error_type, error_message = model_load_error(exc)

    assert error_type == ModelStatusErrorType.ACCESS_REQUIRED
    assert error_message == SAM3_NOT_AUTHORIZED_ERROR_MESSAGE
    assert "Traceback (most recent call last):" not in error_message
    assert not error_message.startswith("OSError:")


def test_traceback_message_prefers_huggingface_token_permission_message_over_terminal_wrapper():
    exc = RuntimeError(
        traceback_message(
            (
                "httpx.HTTPStatusError",
                "Client error '403 Forbidden' for url "
                "'https://huggingface.co/facebook/sam3.1/resolve/main/sam3.1_multiplex.pt'",
            ),
            ("huggingface_hub.errors.HfHubHTTPError", SAM3_TOKEN_PERMISSION_ERROR_MESSAGE),
            ("huggingface_hub.errors.LocalEntryNotFoundError", LOCAL_CACHE_WRAPPER_MESSAGE),
        )
    )

    error_type, error_message = model_load_error(exc)

    assert error_type == ModelStatusErrorType.AUTH_REQUIRED
    assert error_message == SAM3_TOKEN_PERMISSION_ERROR_MESSAGE
    assert "fine-grained token settings" in error_message
    assert "LocalEntryNotFoundError" not in error_message


def test_exception_chain_prefers_huggingface_token_permission_message_over_wrapper():
    exc = RuntimeError(LOCAL_CACHE_WRAPPER_MESSAGE)
    exc.__cause__ = RuntimeError(SAM3_TOKEN_PERMISSION_ERROR_MESSAGE)

    error_type, error_message = model_load_error(exc)

    assert error_type == ModelStatusErrorType.AUTH_REQUIRED
    assert error_message == SAM3_TOKEN_PERMISSION_ERROR_MESSAGE


def test_mixed_access_and_auth_wording_is_classified_as_access_required():
    exc = OSError("Access to model foo is restricted. You must have access to it and be authenticated to access it.")

    error_type, error_message = model_load_error(exc)

    assert error_type == ModelStatusErrorType.ACCESS_REQUIRED
    assert error_message == str(exc)


def test_unknown_failure_is_classified_as_load_failed():
    error_type, error_message = model_load_error(RuntimeError("boom"))

    assert error_type == ModelStatusErrorType.LOAD_FAILED
    assert error_message == "boom"


def test_empty_exception_message_uses_generic_fallback():
    error_type, error_message = model_load_error(RuntimeError())

    assert error_type == ModelStatusErrorType.LOAD_FAILED
    assert "backend logs" in error_message.lower()
