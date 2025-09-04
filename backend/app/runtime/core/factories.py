#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from backend.app.runtime.core.base import StreamReader, StreamWriter, Processor
from .types import IN, OUT
from ..schemas.pipeline import PipelineConfig
from ..schemas.sink import SinkConfig
from ..schemas.source import SourceConfig


class StreamReaderFactory:

    @classmethod
    def create(cls, config: SourceConfig) -> StreamReader:
        pass


class StreamWriterFactory:

    @classmethod
    def create(cls, config: SinkConfig) -> StreamWriter:
        pass


class ProcessorFactory:

    @classmethod
    def create(cls, config: PipelineConfig) -> Processor[IN, OUT]:
        pass
