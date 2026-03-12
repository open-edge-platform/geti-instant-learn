# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import pytest

from runtime.services.license import (
    LicenseConfig,
    LicenseService,
    _get_config_dir,
)


class TestGetConfigDir:
    """Tests for _get_config_dir() helper function."""

    @patch("sys.platform", "linux")
    def test_get_config_dir_linux_with_xdg(self, monkeypatch):
        """On Linux with XDG_CONFIG_HOME set, use that directory."""
        monkeypatch.setenv("XDG_CONFIG_HOME", "/custom/config")
        result = _get_config_dir()
        assert result == Path("/custom/config/instantlearn")

    @patch("sys.platform", "linux")
    def test_get_config_dir_linux_without_xdg(self, monkeypatch):
        """On Linux without XDG_CONFIG_HOME, use ~/.config."""
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        result = _get_config_dir()
        assert result == Path.home() / ".config" / "instantlearn"

    @patch("sys.platform", "darwin")
    def test_get_config_dir_macos(self, monkeypatch):
        """On macOS, use ~/.config (Unix-like behavior)."""
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        result = _get_config_dir()
        assert result == Path.home() / ".config" / "instantlearn"

    @patch("sys.platform", "win32")
    def test_get_config_dir_windows_with_appdata(self, monkeypatch):
        """On Windows with APPDATA set, use that directory."""
        monkeypatch.setenv("APPDATA", "C:\\Users\\Test\\AppData\\Roaming")
        result = _get_config_dir()
        assert str(result) == str(Path("C:\\Users\\Test\\AppData\\Roaming") / "instantlearn")

    @patch("sys.platform", "win32")
    def test_get_config_dir_windows_without_appdata(self, monkeypatch):
        """On Windows without APPDATA, fall back to home directory."""
        monkeypatch.delenv("APPDATA", raising=False)
        result = _get_config_dir()
        expected = Path.home() / "AppData" / "Roaming" / "instantlearn"
        assert result == expected


class TestLicenseConfig:
    """Tests for LicenseConfig dataclass."""

    def test_default_values(self):
        """LicenseConfig has correct default values."""
        config = LicenseConfig()
        assert config.accept_env_var == "INSTANTLEARN_LICENSE_ACCEPTED"
        assert config.consent_file_path == _get_config_dir() / ".license_accepted"

    def test_immutable(self):
        """LicenseConfig is frozen and immutable."""
        config = LicenseConfig()
        with pytest.raises(FrozenInstanceError):
            config.accept_env_var = "DIFFERENT_VAR"

    def test_custom_values(self):
        """LicenseConfig accepts custom values."""
        custom_path = Path("/tmp/custom_license")
        config = LicenseConfig(accept_env_var="CUSTOM_VAR", consent_file_path=custom_path)
        assert config.accept_env_var == "CUSTOM_VAR"
        assert config.consent_file_path == custom_path


class TestLicenseServiceIsAccepted:
    """Tests for the LicenseService.is_accepted() method."""

    def test_is_accepted_via_consent_file(self, tmp_path, monkeypatch):
        """License is accepted when a consent file exists."""
        consent_file = tmp_path / ".license_accepted"
        consent_file.touch()

        with patch("runtime.services.license._get_config_dir", return_value=tmp_path):
            service = LicenseService()
            assert service.is_accepted() is True

    def test_is_accepted_via_env_var(self, tmp_path, monkeypatch):
        """License is accepted when env var is set to 1."""
        monkeypatch.setenv("INSTANTLEARN_LICENSE_ACCEPTED", "1")

        with patch("runtime.services.license._get_config_dir", return_value=tmp_path):
            service = LicenseService()
            assert service.is_accepted() is True

    def test_is_accepted_env_var_wrong_value(self, tmp_path, monkeypatch):
        """License is not accepted when env var has wrong value."""
        monkeypatch.setenv("INSTANTLEARN_LICENSE_ACCEPTED", "aa")

        with patch("runtime.services.license._get_config_dir", return_value=tmp_path):
            service = LicenseService()
            assert service.is_accepted() is False

    def test_is_accepted_env_var_empty(self, tmp_path, monkeypatch):
        """License is not accepted when env var is empty."""
        monkeypatch.setenv("INSTANTLEARN_LICENSE_ACCEPTED", "")

        with patch("runtime.services.license._get_config_dir", return_value=tmp_path):
            service = LicenseService()
            assert service.is_accepted() is False

    def test_is_not_accepted(self, tmp_path, monkeypatch):
        """License is not accepted when neither file nor env var exists."""
        monkeypatch.delenv("INSTANTLEARN_LICENSE_ACCEPTED", raising=False)

        with patch("runtime.services.license._get_config_dir", return_value=tmp_path):
            service = LicenseService()
            assert service.is_accepted() is False

    def test_consent_file_takes_precedence(self, tmp_path, monkeypatch):
        """Consent file is checked before env var."""
        consent_file = tmp_path / ".license_accepted"
        consent_file.touch()
        monkeypatch.delenv("INSTANTLEARN_LICENSE_ACCEPTED", raising=False)

        with patch("runtime.services.license._get_config_dir", return_value=tmp_path):
            service = LicenseService()
            assert service.is_accepted() is True

    def test_env_var_triggers_persistence(self, tmp_path, monkeypatch):
        """Setting env var to 1 attempts to persist acceptance."""
        monkeypatch.setenv("INSTANTLEARN_LICENSE_ACCEPTED", "1")
        consent_file = tmp_path / ".license_accepted"

        with patch("runtime.services.license._get_config_dir", return_value=tmp_path):
            service = LicenseService()
            assert service.is_accepted() is True
            assert consent_file.exists()


class TestLicenseServiceAccept:
    """Tests for LicenseService.accept() method."""

    def test_accept_creates_consent_file(self, tmp_path):
        """Accept creates a consent file with timestamp."""
        with patch("runtime.services.license._get_config_dir", return_value=tmp_path):
            service = LicenseService()
            service.accept()

        consent_file = tmp_path / ".license_accepted"
        assert consent_file.exists()
        content = consent_file.read_text(encoding="utf-8")
        assert "License accepted at" in content
        # Verify ISO format timestamp
        assert "T" in content and ("Z" in content or "+" in content or "-" in content[-6:])

    def test_accept_creates_parent_directories(self, tmp_path):
        """Accept creates parent directories if they don't exist."""
        nested_path = tmp_path / "config" / "instantlearn"

        with patch("runtime.services.license._get_config_dir", return_value=nested_path):
            service = LicenseService()
            service.accept()

        consent_file = nested_path / ".license_accepted"
        assert consent_file.exists()
        assert consent_file.parent.exists()

    @patch("sys.platform", "linux")
    def test_accept_sets_file_permissions_unix(self, tmp_path):
        """Accept sets restrictive permissions on Unix-like systems."""
        with patch("runtime.services.license._get_config_dir", return_value=tmp_path):
            service = LicenseService()
            service.accept()

        consent_file = tmp_path / ".license_accepted"
        # Check that permissions are owner read/write only (0o600)
        assert consent_file.stat().st_mode & 0o777 == 0o600

    @patch("sys.platform", "win32")
    def test_accept_skips_chmod_on_windows(self, tmp_path):
        """Accept skips chmod on Windows."""
        with patch("runtime.services.license._get_config_dir", return_value=tmp_path):
            service = LicenseService()
            service.accept()

        consent_file = tmp_path / ".license_accepted"
        assert consent_file.exists()
        # Just verify it doesn't raise an error

    def test_accept_permission_error(self, tmp_path):
        """Accept raises OSError when file creation fails."""
        with patch("runtime.services.license._get_config_dir", return_value=tmp_path):
            service = LicenseService()

            with patch.object(Path, "write_text", side_effect=PermissionError("Access denied")):
                with pytest.raises(OSError, match="Cannot create license consent file"):
                    service.accept()

    def test_accept_idempotent(self, tmp_path):
        """Accept can be called multiple times without error."""
        with patch("runtime.services.license._get_config_dir", return_value=tmp_path):
            service = LicenseService()
            service.accept()
            service.accept()  # the second call should not raise

        consent_file = tmp_path / ".license_accepted"
        assert consent_file.exists()


class TestLicenseServiceTryPersistAcceptance:
    """Tests for LicenseService._try_persist_acceptance() method."""

    def test_try_persist_creates_file(self, tmp_path):
        """Try persist creates consent file when it doesn't exist."""
        with patch("runtime.services.license._get_config_dir", return_value=tmp_path):
            service = LicenseService()
            service._try_persist_acceptance()

        consent_file = tmp_path / ".license_accepted"
        assert consent_file.exists()

    def test_try_persist_skips_if_exists(self, tmp_path):
        """Try persist does nothing if consent file already exists."""
        consent_file = tmp_path / ".license_accepted"
        consent_file.write_text("existing content", encoding="utf-8")
        original_mtime = consent_file.stat().st_mtime

        with patch("runtime.services.license._get_config_dir", return_value=tmp_path):
            service = LicenseService()
            service._try_persist_acceptance()

        # File should not be modified
        assert consent_file.stat().st_mtime == original_mtime
        assert consent_file.read_text(encoding="utf-8") == "existing content"

    def test_try_persist_handles_oserror_gracefully(self, tmp_path):
        """Try persist logs warning but doesn't raise on OSError."""
        with patch("runtime.services.license._get_config_dir", return_value=tmp_path):
            service = LicenseService()

            with patch.object(service, "accept", side_effect=OSError("Cannot write")):
                # Should not raise
                service._try_persist_acceptance()

        # Consent file should not exist
        consent_file = tmp_path / ".license_accepted"
        assert not consent_file.exists()


class TestLicenseServiceIntegration:
    """Integration tests for LicenseService."""

    def test_full_workflow_env_var_to_file(self, tmp_path, monkeypatch):
        """Complete workflow: env var triggers acceptance and file creation."""
        monkeypatch.setenv("INSTANTLEARN_LICENSE_ACCEPTED", "1")

        with patch("runtime.services.license._get_config_dir", return_value=tmp_path):
            service = LicenseService()

            # First check triggers file creation
            assert service.is_accepted() is True

            # File should now exist
            consent_file = tmp_path / ".license_accepted"
            assert consent_file.exists()

            # Remove env var
            monkeypatch.delenv("INSTANTLEARN_LICENSE_ACCEPTED")

            # Should still be accepted via file
            assert service.is_accepted() is True

    def test_multiple_services_share_state(self, tmp_path):
        """Multiple LicenseService instances share the same consent file."""
        with patch("runtime.services.license._get_config_dir", return_value=tmp_path):
            service1 = LicenseService()
            service2 = LicenseService()

            assert service1.is_accepted() is False
            assert service2.is_accepted() is False

            service1.accept()

            assert service1.is_accepted() is True
            assert service2.is_accepted() is True
