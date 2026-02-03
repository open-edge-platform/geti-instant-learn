# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from domain.errors import ResourceConnectionError, ResourceType
from domain.services.schemas.writer import WriterConfig
from domain.services.validators.sink_connection import SinkConnectionValidator
from runtime.core.components.factories.writer import StreamWriterFactory

logger = logging.getLogger(__name__)


class RuntimeSinkConnectionValidator(SinkConnectionValidator):
    """Runtime validator that checks connectivity via StreamWriter."""

    def validate(self, config: WriterConfig, sink_id: UUID | None) -> None:
        writer = StreamWriterFactory.create(config)
        try:
            writer.connect()
        except ConnectionError as exc:
            logger.error("Sink connection validation failed: %s", exc)
            raise ResourceConnectionError(
                resource_type=ResourceType.SINK,
                resource_id=str(sink_id) if sink_id else None,
                message=str(exc),
            ) from exc
        finally:
            writer.close()
