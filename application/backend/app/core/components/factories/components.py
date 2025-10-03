#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from queue import Queue
from typing import Any

from core.components.broadcaster import FrameBroadcaster
from core.components.factories.model import ModelFactory
from core.components.factories.reader import StreamReaderFactory
from core.components.factories.writer import StreamWriterFactory
from core.components.processor import Processor
from core.components.sink import Sink
from core.components.source import Source


class ComponentFactory(ABC):
    @abstractmethod
    def create_source(self, in_queue: Queue, reader_conf: Any) -> Source: ...

    @abstractmethod
    def create_processor(self, in_queue: Queue, broadcaster: FrameBroadcaster, model_config: Any) -> Processor: ...

    @abstractmethod
    def create_sink(self, broadcaster: FrameBroadcaster, writer_conf: Any) -> Sink: ...


class DefaultComponentFactory(ComponentFactory):
    def create_source(self, in_queue: Queue, reader_conf: Any) -> Source:
        return Source(in_queue, StreamReaderFactory.create(reader_conf))

    def create_processor(self, in_queue: Queue, broadcaster: FrameBroadcaster, model_config: Any) -> Processor:
        return Processor(in_queue, broadcaster, ModelFactory.create(model_config))

    def create_sink(self, broadcaster: FrameBroadcaster, writer_conf: Any) -> Sink:
        return Sink(broadcaster, StreamWriterFactory.create(writer_conf))
