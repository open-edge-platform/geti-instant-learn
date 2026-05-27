# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from domain.services.schemas.model_status import ModelStatusErrorType
from runtime.services.model_load_error import build_model_load_error


def test_wrapped_huggingface_auth_failure_is_classified_as_auth_required():
    exc = OSError(
        "You are trying to access a gated repo. Make sure to have access to it at "
        "https://huggingface.co/facebook/sam3.1. Please log in."
    )

    error_type, error_message = build_model_load_error(exc)

    assert error_type == ModelStatusErrorType.AUTH_REQUIRED
    assert "hf auth login" in error_message


def test_wrapped_huggingface_access_failure_is_classified_as_access_required():
    exc = OSError(
        "Cannot access gated repo for url https://huggingface.co/facebook/sam3.1/resolve/main/tokenizer_config.json. "
        "Access to model facebook/sam3.1 is restricted and you are not in the authorized list. "
        "Visit https://huggingface.co/facebook/sam3.1 to ask for access."
    )

    error_type, error_message = build_model_load_error(exc)

    assert error_type == ModelStatusErrorType.ACCESS_REQUIRED
    assert "request access" in error_message.lower()
    assert "hf auth login" not in error_message


def test_huggingface_value_error_access_failure_is_classified_as_access_required():
    exc = ValueError(
        "User does not have access to the weights of the DinoV3 model.\n"
        "Please follow these steps:\n"
        "1. Request access on the HuggingFace website: "
        "https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m\n"
        "2. Set your HuggingFace credentials using one of these methods:\n"
        "   - Run: hf auth login\n"
        "   - Set environment variable: export HUGGINGFACE_HUB_TOKEN=your_token"
    )

    error_type, error_message = build_model_load_error(exc)

    assert error_type == ModelStatusErrorType.ACCESS_REQUIRED
    assert "request access" in error_message.lower()
    assert "hf auth login" not in error_message


def test_mixed_access_and_auth_wording_is_classified_as_access_required():
    exc = OSError("Access to model foo is restricted. You must have access to it and be authenticated to access it.")

    error_type, error_message = build_model_load_error(exc)

    assert error_type == ModelStatusErrorType.ACCESS_REQUIRED
    assert "request access" in error_message.lower()
    assert "hf auth login" not in error_message


def test_unknown_failure_is_classified_as_load_failed():
    error_type, error_message = build_model_load_error(RuntimeError("boom"))

    assert error_type == ModelStatusErrorType.LOAD_FAILED
    assert "backend logs" in error_message.lower()
