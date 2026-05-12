# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from domain.services.schemas.reader import ReaderConfig
from runtime.core.components.factories.reader import StreamReaderFactory

logger = logging.getLogger(__name__)


class ReaderConfigValidator:
    """Validator that checks a reader configuration is usable."""

    def __init__(self, reader_factory: StreamReaderFactory | None = None) -> None:
        self._reader_factory = reader_factory or StreamReaderFactory()

    def validate(self, config: ReaderConfig) -> None:
        with self._reader_factory.create(config) as reader:
            reader.validate_config()
