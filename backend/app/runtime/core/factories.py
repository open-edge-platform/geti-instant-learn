#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from backend.app.runtime.core.base import StreamReader, StreamWriter, Processor
from ..schemas.pipeline import PipelineConfig, InputData, OutputData
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
    def create(cls, config: PipelineConfig) -> Processor[InputData, OutputData]:
        pass
