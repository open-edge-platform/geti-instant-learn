#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any

from runtime.core.components.broadcaster import FrameBroadcaster
from runtime.core.components.factories.model import ModelFactory
from runtime.core.components.factories.reader import StreamReaderFactory
from runtime.core.components.factories.writer import StreamWriterFactory
from runtime.core.components.processor import Processor
from runtime.core.components.schemas.processor import InputData, OutputData
from runtime.core.components.sink import Sink
from runtime.core.components.source import Source


class ComponentFactory(ABC):
    @abstractmethod
    def create_source(self, reader_conf: Any, inbound_broadcaster: FrameBroadcaster[InputData]) -> Source: ...

    @abstractmethod
    def create_processor(
        self,
        inbound_broadcaster: FrameBroadcaster[InputData],
        outbound_broadcaster: FrameBroadcaster[OutputData],
        model_config: Any,
    ) -> Processor: ...

    @abstractmethod
    def create_sink(self, outbound_broadcaster: FrameBroadcaster[OutputData], writer_conf: Any) -> Sink: ...


class DefaultComponentFactory(ComponentFactory):
    def create_source(self, reader_conf: Any, inbound_broadcaster: FrameBroadcaster[InputData]) -> Source:
        return Source(StreamReaderFactory.create(reader_conf), inbound_broadcaster)

    def create_processor(
        self,
        inbound_broadcaster: FrameBroadcaster[InputData],
        outbound_broadcaster: FrameBroadcaster[OutputData],
        model_config: Any,
    ) -> Processor:
        return Processor(inbound_broadcaster, outbound_broadcaster, ModelFactory.create(model_config))

    def create_sink(self, outbound_broadcaster: FrameBroadcaster[OutputData], writer_conf: Any) -> Sink:
        return Sink(outbound_broadcaster, StreamWriterFactory.create(writer_conf))
