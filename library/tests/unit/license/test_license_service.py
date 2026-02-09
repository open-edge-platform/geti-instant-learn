# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LicenseService."""

from pathlib import Path
from unittest.mock import patch

import pytest

from instantlearn_license.service import LicenseConfig, LicenseNotAcceptedError, LicenseService


@pytest.fixture
def temp_consent_file(tmp_path: Path) -> Path:
    return tmp_path / ".license_accepted"


@pytest.fixture
def license_service(temp_consent_file: Path) -> LicenseService:
    service = LicenseService()
    service._config = LicenseConfig(consent_file_path=temp_consent_file)
    return service


@pytest.fixture
def clean_env():
    """Fixture to provide clean environment for tests."""
    with patch.dict("os.environ", {}, clear=True):
        yield


def test_config_default_values() -> None:
    config = LicenseConfig()

    assert config.accept_env_var == "INSTANTLEARN_LICENSE_ACCEPTED"
    assert config.skip_check_env_var == "INSTANTLEARN_SKIP_LICENSE_CHECK_ON_IMPORT"
    assert config.consent_file_path.name == ".license_accepted"


def test_not_accepted_by_default(license_service: LicenseService, clean_env) -> None:
    assert license_service.is_accepted() is False


def test_accepted_via_consent_file(
    license_service: LicenseService, temp_consent_file: Path, clean_env
) -> None:
    temp_consent_file.parent.mkdir(parents=True, exist_ok=True)
    temp_consent_file.write_text("License accepted")

    assert license_service.is_accepted() is True


def test_accepted_via_env_var_and_persists(
    license_service: LicenseService, temp_consent_file: Path
) -> None:
    with patch.dict("os.environ", {"INSTANTLEARN_LICENSE_ACCEPTED": "1"}, clear=True):
        assert license_service.is_accepted() is True

    assert temp_consent_file.exists()


@pytest.mark.parametrize("value", ["0", "true", "yes", ""])
def test_env_var_requires_exact_value_1(
    license_service: LicenseService, value: str
) -> None:
    with patch.dict("os.environ", {"INSTANTLEARN_LICENSE_ACCEPTED": value}, clear=True):
        assert license_service.is_accepted() is False


def test_accept_creates_consent_file(
    license_service: LicenseService, temp_consent_file: Path
) -> None:
    license_service.accept()

    assert temp_consent_file.exists()
    assert "License accepted at" in temp_consent_file.read_text()


def test_accept_creates_nested_directories(tmp_path: Path) -> None:
    nested_path = tmp_path / "deep" / "nested" / ".license_accepted"
    service = LicenseService()
    service._config = LicenseConfig(consent_file_path=nested_path)

    service.accept()

    assert nested_path.exists()


def test_accept_raises_on_permission_error(license_service: LicenseService) -> None:
    with patch("pathlib.Path.mkdir", side_effect=PermissionError("denied")):
        with pytest.raises(OSError, match="Cannot create license consent file"):
            license_service.accept()


def test_skip_env_var_bypasses_all_checks(license_service: LicenseService) -> None:
    with patch.dict(
        "os.environ", {"INSTANTLEARN_SKIP_LICENSE_CHECK_ON_IMPORT": "1"}, clear=True
    ):
        with patch.object(license_service, "is_accepted") as mock:
            license_service.require_accepted()
            mock.assert_not_called()


def test_require_accepted_passes_when_already_accepted(
    license_service: LicenseService, temp_consent_file: Path, clean_env
) -> None:
    temp_consent_file.parent.mkdir(parents=True, exist_ok=True)
    temp_consent_file.write_text("License accepted")

    license_service.require_accepted()


def test_require_accepted_raises_in_non_interactive_mode(
    license_service: LicenseService, clean_env
) -> None:
    with pytest.raises(LicenseNotAcceptedError):
        license_service.require_accepted(interactive=False)


@pytest.mark.parametrize(
    "user_input,should_accept",
    [("y", True), ("n", False)],
)
def test_interactive_prompt_response(
    license_service: LicenseService,
    temp_consent_file: Path,
    clean_env,
    user_input: str,
    should_accept: bool,
) -> None:
    with patch("builtins.input", return_value=user_input):
        if should_accept:
            license_service.require_accepted(interactive=True)
            assert temp_consent_file.exists()
        else:
            with pytest.raises(LicenseNotAcceptedError, match="License not accepted"):
                license_service.require_accepted(interactive=True)


def test_eof_on_input_raises_error(license_service: LicenseService, clean_env) -> None:
    with patch("builtins.input", side_effect=EOFError):
        with pytest.raises(LicenseNotAcceptedError, match="Could not read input"):
            license_service.require_accepted(interactive=True)


def test_auto_detects_interactive_mode(license_service: LicenseService, clean_env) -> None:
    with patch.object(license_service, "_is_interactive", return_value=False) as mock:
        with pytest.raises(LicenseNotAcceptedError):
            license_service.require_accepted()
        mock.assert_called_once()


def test_not_interactive_when_no_tty(license_service: LicenseService) -> None:
    with patch("sys.stdin.isatty", return_value=False):
        assert license_service._is_interactive() is False


@pytest.mark.parametrize(
    "ci_var", ["CI", "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_URL", "KUBERNETES_SERVICE_HOST"]
)
def test_not_interactive_in_ci_environments(license_service: LicenseService, ci_var: str) -> None:
    with patch("sys.stdin.isatty", return_value=True):
        with patch.dict("os.environ", {ci_var: "true"}, clear=True):
            assert license_service._is_interactive() is False


def test_interactive_with_tty_and_no_ci(license_service: LicenseService) -> None:
    with patch("sys.stdin.isatty", return_value=True):
        with patch.dict("os.environ", {}, clear=True):
            assert license_service._is_interactive() is True


def test_is_jupyter_notebook_returns_true_for_zmq_shell(license_service: LicenseService) -> None:
    mock_ipython = type("MockIPython", (), {"__class__": type("ZMQInteractiveShell", (), {})})()

    with patch("instantlearn_license.service.get_ipython", return_value=mock_ipython, create=True):
        assert license_service._is_jupyter_notebook() is True


def test_is_jupyter_notebook_returns_false_for_terminal_ipython(license_service: LicenseService) -> None:
    mock_ipython = type("MockIPython", (), {"__class__": type("TerminalInteractiveShell", (), {})})()

    with patch("instantlearn_license.service.get_ipython", return_value=mock_ipython, create=True):
        assert license_service._is_jupyter_notebook() is False


def test_is_jupyter_notebook_returns_false_when_not_in_ipython(license_service: LicenseService) -> None:
    with patch("instantlearn_license.service.get_ipython", side_effect=NameError, create=True):
        assert license_service._is_jupyter_notebook() is False


def test_interactive_in_jupyter_notebook(license_service: LicenseService) -> None:
    with patch.object(license_service, "_is_jupyter_notebook", return_value=True):
        with patch("sys.stdin.isatty", return_value=False):
            assert license_service._is_interactive() is True


def test_jupyter_notebook_can_accept_license(
        license_service: LicenseService, temp_consent_file: Path, clean_env) -> None:
    with patch.object(license_service, "_is_jupyter_notebook", return_value=True):
        with patch("sys.stdin.isatty", return_value=False):
            with patch("builtins.input", return_value="y"):
                license_service.require_accepted()
                assert temp_consent_file.exists()


def test_error_default_message_includes_instructions() -> None:
    error = LicenseNotAcceptedError()

    assert "INSTANTLEARN_LICENSE_ACCEPTED=1" in str(error)
    assert "instantlearn" in str(error)


def test_error_custom_message() -> None:
    assert str(LicenseNotAcceptedError("Custom error")) == "Custom error"
