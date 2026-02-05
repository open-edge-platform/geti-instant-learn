# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LICENSE_MESSAGE = """This software is subject to additional third-party licenses. By using it, you agree to:
- [SAM3 License Agreement](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
- [DINOv3 License Agreement](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)

By using the library I acknowledge I have:
      - read and understood the license terms at the links above;
      - confirmed the linked terms govern the contents I seek to access and use; and
      - accepted and agreed to the linked license terms."""


class LicenseNotAcceptedError(Exception):
    """Raised when the license has not been accepted. This exception is raised in non-interactive contexts."""

    def __init__(self, message: str | None = None) -> None:
        """Initialize the exception with the license message.

        Args:
            message: Optional custom message. If not provided, uses the default license message with instructions.
        """
        if message is None:
            message = (
                f"{LICENSE_MESSAGE}\n\n"
                "To accept the license, either:\n"
                "  - Set environment variable: `INSTANTLEARN_LICENSE_ACCEPTED=1`, or\n"
                "  - Run `instantlearn` in an interactive terminal and confirm the prompt\n"
            )
        super().__init__(message)


def _get_config_dir() -> Path:
    """Get the platform-appropriate config directory.

    Returns:
        Path to the config directory (~/.config/instantlearn on Unix, %APPDATA%/instantlearn on Windows).
    """
    if sys.platform == "win32":
        # Windows: use APPDATA or fall back to home
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "instantlearn"
        return Path.home() / "AppData" / "Roaming" / "instantlearn"

    # Unix-like (Linux, macOS): use XDG_CONFIG_HOME or ~/.config
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config) / "instantlearn"
    return Path.home() / ".config" / "instantlearn"


@dataclass(frozen=True)
class LicenseConfig:
    """Configuration for license acceptance checking.

    Attributes:
        accept_env_var: Name of the environment variable that indicates license acceptance when set to "1".
        skip_check_env_var: Name of the environment variable that skips license check on import when set to "1".
        consent_file_path: Path to the file that indicates license acceptance when it exists.
    """

    accept_env_var: str = "INSTANTLEARN_LICENSE_ACCEPTED"
    skip_check_env_var: str = "INSTANTLEARN_SKIP_LICENSE_CHECK_ON_IMPORT"
    consent_file_path: Path = field(default_factory=lambda: _get_config_dir() / ".license_accepted")


class LicenseService:
    """Service for managing license acceptance."""

    def __init__(self) -> None:
        """Initialize the license service."""
        self._config = LicenseConfig()

    def is_accepted(self) -> bool:
        """Check if the license has been accepted."""
        if self._config.consent_file_path.exists():
            logger.debug("License accepted via consent file %s", self._config.consent_file_path)
            return True

        env_value = os.environ.get(self._config.accept_env_var, "")
        if env_value == "1":
            logger.debug("License accepted via environment variable %s", self._config.accept_env_var)
            self._try_persist_acceptance()
            return True

        return False

    def _try_persist_acceptance(self) -> None:
        """Attempt to persist acceptance to consent file without raising errors."""
        if self._config.consent_file_path.exists():
            return
        try:
            self.accept()
        except OSError as e:
            logger.warning("Failed to create license consent file: %s", e)


    def accept(self) -> None:
        """Accept the license by creating the consent file."""
        consent_path = self._config.consent_file_path
        try:
            consent_path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(tz=timezone.utc).isoformat()
            content = f"License accepted at {timestamp}\n"
            consent_path.write_text(content, encoding="utf-8")
        except PermissionError as e:
            raise OSError(f"Cannot create license consent file: {e}") from e

        if sys.platform != "win32":
            consent_path.chmod(0o600)  # owner read/write only

        logger.info("License accepted. Consent file created at %s", consent_path)

    @staticmethod
    def _is_interactive() -> bool:
        """Determine if the current context is interactive.

        Checks multiple indicators:
        - stdin is a TTY
        - Not running in common CI environments
        - Not running under common non-interactive indicators
        """
        if not sys.stdin.isatty():
            return False

        non_interactive_indicators = [
            "CI",
            "CONTINUOUS_INTEGRATION",
            "GITHUB_ACTIONS",
            "GITLAB_CI",
            "JENKINS_URL",
            "TRAVIS",
            "CIRCLECI",
            "BUILDKITE",
            "TF_BUILD",
            "KUBERNETES_SERVICE_HOST",
        ]
        for indicator in non_interactive_indicators:
            if os.environ.get(indicator) is not None:
                logger.debug("Non-interactive environment detected (%s)", indicator)
                return False

        return True

    def require_accepted(self, interactive: bool | None = None) -> None:
        """Require that the license has been accepted.

        If INSTANTLEARN_SKIP_LICENSE_CHECK_ON_IMPORT is set to "1", this method
        returns immediately without checking or prompting.

        If the license is not accepted:
        - In interactive mode: displays the license message via logging
          and prompts the user to type 'y' to accept.
        - In non-interactive mode: raises LicenseNotAcceptedError.

        Args:
            interactive: If True, always treat as interactive. If False,
                always treat as non-interactive. If None (default), auto-detect
                based on TTY and environment.

        Raises:
            LicenseNotAcceptedError: If the license is not accepted and the
                context is non-interactive, or if the user declines.
        """
        if os.environ.get(self._config.skip_check_env_var, "") == "1":
            return

        if self.is_accepted():
            return

        if interactive is None:
            interactive = self._is_interactive()

        if not interactive:
            logger.error("License was not accepted.")
            raise LicenseNotAcceptedError

        # Interactive mode: show license and prompt for acceptance
        logger.warning("\n%s\n", LICENSE_MESSAGE)
        try:
            response = input("\nDo you accept the license terms? [y/N]: ").strip().lower()
        except EOFError:
            raise LicenseNotAcceptedError(
                f"{LICENSE_MESSAGE}\n\nCould not read input. Please set INSTANTLEARN_LICENSE_ACCEPTED=1 to accept.",
            ) from None

        if response == "y":
            self.accept()
        else:
            raise LicenseNotAcceptedError(
                "License not accepted. You must accept the license terms to use Geti Instant Learn library.",
            )
