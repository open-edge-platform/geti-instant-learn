# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from domain.services.schemas.writer import WriterConfig
from runtime.core.components.factories.writer import StreamWriterFactory
from runtime.errors import SinkConnectionError

logger = logging.getLogger(__name__)


class SinkConnectionValidator:
    """Validator that checks connectivity via StreamWriter."""

    def validate(self, config: WriterConfig) -> None:
        try:
            with StreamWriterFactory.create(config) as writer:
                writer.connect()
        except ConnectionError as exc:
            logger.error("Sink connection validation failed: %s", exc)
            raise SinkConnectionError(str(exc)) from exc
