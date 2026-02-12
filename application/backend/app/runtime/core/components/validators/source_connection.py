# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from domain.services.schemas.reader import ReaderConfig
from runtime.core.components.factories.reader import StreamReaderFactory
from runtime.errors import SourceConnectionError

logger = logging.getLogger(__name__)


class SourceConnectionValidator:
    """Validator that checks connectivity via StreamReader."""

    def validate(self, config: ReaderConfig) -> None:
        """
        Validate that a source can be connected to.

        Args:
            config: The reader configuration to validate

        Raises:
            SourceConnectionError: If the source cannot be connected to
        """
        try:
            reader = StreamReaderFactory.create(config)
            # Try to connect using context manager to ensure cleanup
            with reader:
                reader.connect()
        except (RuntimeError, ConnectionError, OSError) as exc:
            logger.error("Source connection validation failed: %s", exc)
            raise SourceConnectionError(str(exc)) from exc
