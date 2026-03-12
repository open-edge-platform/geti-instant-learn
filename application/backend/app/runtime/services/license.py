# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys
from datetime import UTC, datetime

from settings import get_settings

logger = logging.getLogger(__name__)


LICENSE_MESSAGE = """This software is subject to additional third-party licenses. By using it, you agree to:
- [SAM3 License Agreement](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
- [DINOv3 License Agreement](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)

By using Geti Instant Learn library I acknowledge I have:
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
        settings = get_settings()
        if message is None:
            message = (
                f"{LICENSE_MESSAGE}\n\n"
                f"To accept the license, set the environment variable `{settings.license_accept_env_var}=1`"
            )
        super().__init__(message)


class LicenseService:
    """Service for managing license acceptance."""

    def __init__(self) -> None:
        """Initialize the license service."""
        self._settings = get_settings()

    def is_accepted(self) -> bool:
        """Check if the license has been accepted."""
        if self._settings.license_consent_file_path.exists():
            logger.debug("License accepted via consent file %s", self._settings.license_consent_file_path)
            return True

        env_value = os.environ.get(self._settings.license_accept_env_var, "")
        if env_value == "1":
            logger.debug("License accepted via environment variable %s", self._settings.license_accept_env_var)
            return True

        return False

    def accept(self) -> None:
        """Accept the license by creating the consent file."""
        consent_path = self._settings.license_consent_file_path
        try:
            consent_path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(tz=UTC).isoformat()
            content = f"License accepted at {timestamp}\n"
            consent_path.write_text(content, encoding="utf-8")
        except PermissionError as e:
            raise OSError(f"Cannot create license consent file: {e}") from e

        if sys.platform != "win32":
            consent_path.chmod(0o600)  # owner read/write only

        logger.info("License accepted. Consent file created at %s", consent_path)
