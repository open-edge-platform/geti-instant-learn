# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch

import pytest

from runtime.services.license import LicenseService
from settings import Settings


@pytest.fixture
def mock_config_dir(tmp_path):
    """Mock Settings.config_dir to use a temporary directory."""
    with patch.object(Settings, "config_dir", tmp_path):
        yield tmp_path


@pytest.fixture
def license_service(mock_config_dir):
    """Create a LicenseService instance with mocked config directory."""
    return LicenseService()


@pytest.fixture
def consent_file_path(mock_config_dir):
    """Get path to the consent file in the mocked config directory."""
    return mock_config_dir / ".license_accepted"


@pytest.fixture
def clean_env(monkeypatch):
    """Remove license acceptance environment variable."""
    monkeypatch.delenv("INSTANTLEARN_LICENSE_ACCEPTED", raising=False)
    yield monkeypatch


class TestSettingsConfigDir:
    """Tests for Settings.config_dir property."""

    @patch("sys.platform", "linux")
    def test_config_dir_linux_with_xdg(self, monkeypatch):
        """On Linux with XDG_CONFIG_HOME set, use that directory."""
        monkeypatch.setenv("XDG_CONFIG_HOME", "/custom/config")
        settings = Settings()
        assert settings.config_dir == Path("/custom/config/instantlearn")

    @patch("sys.platform", "linux")
    def test_config_dir_linux_without_xdg(self, monkeypatch):
        """On Linux without XDG_CONFIG_HOME, use ~/.config."""
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        settings = Settings()
        assert settings.config_dir == Path.home() / ".config" / "instantlearn"

    @patch("sys.platform", "darwin")
    def test_config_dir_macos(self, monkeypatch):
        """On macOS, use ~/.config (Unix-like behavior)."""
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        settings = Settings()
        assert settings.config_dir == Path.home() / ".config" / "instantlearn"

    @patch("sys.platform", "win32")
    def test_config_dir_windows_with_appdata(self, monkeypatch):
        """On Windows with APPDATA set, use that directory."""
        monkeypatch.setenv("APPDATA", "C:\\Users\\Test\\AppData\\Roaming")
        settings = Settings()
        assert str(settings.config_dir) == str(Path("C:\\Users\\Test\\AppData\\Roaming") / "instantlearn")

    @patch("sys.platform", "win32")
    def test_config_dir_windows_without_appdata(self, monkeypatch):
        """On Windows without APPDATA, fall back to home directory."""
        monkeypatch.delenv("APPDATA", raising=False)
        settings = Settings()
        expected = Path.home() / "AppData" / "Roaming" / "instantlearn"
        assert settings.config_dir == expected


class TestSettingsLicenseProperties:
    """Tests for license-related Settings properties."""

    def test_license_accept_env_var_default(self):
        """Settings has correct default value for license_accept_env_var."""
        settings = Settings()
        assert settings.license_accept_env_var == "INSTANTLEARN_LICENSE_ACCEPTED"

    def test_license_consent_file_path(self):
        """Settings.license_consent_file_path returns correct path."""
        with patch.object(Settings, "config_dir", Path("/test/config")):
            settings = Settings()
            assert settings.license_consent_file_path == Path("/test/config/.license_accepted")


class TestLicenseServiceIsAccepted:
    """Tests for the LicenseService.is_accepted() method."""

    def test_is_accepted_via_consent_file(self, license_service, consent_file_path, clean_env):
        """License is accepted when a consent file exists."""
        consent_file_path.touch()
        assert license_service.is_accepted() is True

    def test_is_accepted_via_env_var(self, license_service, monkeypatch):
        """License is accepted when env var is set to 1."""
        monkeypatch.setenv("INSTANTLEARN_LICENSE_ACCEPTED", "1")
        assert license_service.is_accepted() is True

    def test_is_accepted_env_var_wrong_value(self, license_service, monkeypatch):
        """License is not accepted when env var has wrong value."""
        monkeypatch.setenv("INSTANTLEARN_LICENSE_ACCEPTED", "aa")
        assert license_service.is_accepted() is False

    def test_is_accepted_env_var_empty(self, license_service, monkeypatch):
        """License is not accepted when env var is empty."""
        monkeypatch.setenv("INSTANTLEARN_LICENSE_ACCEPTED", "")
        assert license_service.is_accepted() is False

    def test_is_not_accepted(self, license_service, clean_env):
        """License is not accepted when neither file nor env var exists."""
        assert license_service.is_accepted() is False

    def test_consent_file_takes_precedence(self, license_service, consent_file_path, clean_env):
        """Consent file is checked before env var."""
        consent_file_path.touch()
        assert license_service.is_accepted() is True

    def test_env_var_does_not_create_file(self, license_service, consent_file_path, monkeypatch):
        """Setting env var to 1 does not create consent file."""
        monkeypatch.setenv("INSTANTLEARN_LICENSE_ACCEPTED", "1")
        assert license_service.is_accepted() is True
        assert not consent_file_path.exists()


class TestLicenseServiceAccept:
    """Tests for LicenseService.accept() method."""

    def test_accept_creates_consent_file(self, license_service, consent_file_path):
        """Accept creates a consent file with timestamp."""
        license_service.accept()

        assert consent_file_path.exists()
        content = consent_file_path.read_text(encoding="utf-8")
        assert "License accepted at" in content
        assert "T" in content and ("Z" in content or "+" in content or "-" in content[-6:])

    def test_accept_creates_parent_directories(self, tmp_path):
        """Accept creates parent directories if they don't exist."""
        nested_path = tmp_path / "config" / "instantlearn"

        with patch.object(Settings, "config_dir", nested_path):
            service = LicenseService()
            service.accept()

        consent_file = nested_path / ".license_accepted"
        assert consent_file.exists()
        assert consent_file.parent.exists()

    @patch("sys.platform", "linux")
    def test_accept_sets_file_permissions_unix(self, license_service, consent_file_path):
        """Accept sets restrictive permissions on Unix-like systems."""
        license_service.accept()
        assert consent_file_path.stat().st_mode & 0o777 == 0o600

    @patch("sys.platform", "win32")
    def test_accept_skips_chmod_on_windows(self, license_service, consent_file_path):
        """Accept skips chmod on Windows."""
        license_service.accept()
        assert consent_file_path.exists()

    def test_accept_permission_error(self, license_service):
        """Accept raises OSError when file creation fails."""
        with patch.object(Path, "write_text", side_effect=PermissionError("Access denied")):
            with pytest.raises(OSError, match="Cannot create license consent file"):
                license_service.accept()

    def test_accept_idempotent(self, license_service, consent_file_path):
        """Accept can be called multiple times without error."""
        license_service.accept()
        license_service.accept()
        assert consent_file_path.exists()


class TestLicenseServiceIntegration:
    """Integration tests for LicenseService."""

    def test_env_var_does_not_trigger_file_creation(self, license_service, consent_file_path, monkeypatch):
        """Env var does NOT trigger file creation."""
        monkeypatch.setenv("INSTANTLEARN_LICENSE_ACCEPTED", "1")
        assert license_service.is_accepted() is True
        assert not consent_file_path.exists()

    def test_explicit_accept_creates_file(self, license_service, consent_file_path, clean_env):
        """Explicit accept() call creates file regardless of env var."""
        assert license_service.is_accepted() is False

        license_service.accept()

        assert license_service.is_accepted() is True
        assert consent_file_path.exists()

    def test_multiple_services_share_state(self, mock_config_dir):
        """Multiple LicenseService instances share the same consent file."""
        service1 = LicenseService()
        service2 = LicenseService()

        assert service1.is_accepted() is False
        assert service2.is_accepted() is False

        service1.accept()

        assert service1.is_accepted() is True
        assert service2.is_accepted() is True
