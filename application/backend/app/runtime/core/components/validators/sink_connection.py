# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from domain.services.schemas.writer import WriterConfig
from runtime.core.components.factories.writer import StreamWriterFactory
from runtime.errors import SinkConnectionError

logger = logging.getLogger(__name__)


class RuntimeSinkConnectionValidator:
    """Runtime validator that checks connectivity via StreamWriter."""

    def validate(self, config: WriterConfig) -> None:
        writer = StreamWriterFactory.create(config)
        try:
            writer.connect()
        except ConnectionError as exc:
            logger.error("Sink connection validation failed: %s", exc)
            raise SinkConnectionError(str(exc)) from exc
        finally:
            writer.close()
