#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from backend.app.runtime.core.base import StreamReader, StreamWriter, Processor
from backend.app.schemas.processor import ProcessorConfig
from backend.app.schemas.sink import SinkConfig
from backend.app.schemas.source import SourceConfig
from .types import IN, OUT


class StreamReaderFactory:
    """Abstract factory for creating StreamReader instances from a configuration."""

    @classmethod
    def create(cls, config: SourceConfig) -> StreamReader:
        pass


class StreamWriterFactory:
    """Abstract factory for creating StreamWriter instances from a configuration."""

    @classmethod
    def create(cls, config: SinkConfig) -> StreamWriter:
        pass


class ProcessorFactory:
    """Abstract factory for creating Processor instances from a configuration."""

    @classmethod
    def create(cls, config: ProcessorConfig) -> Processor[IN, OUT]:
        pass
