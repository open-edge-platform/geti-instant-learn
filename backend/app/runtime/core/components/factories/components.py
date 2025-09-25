#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from queue import Queue
from typing import Any

from runtime.core.components.broadcaster import FrameBroadcaster
from runtime.core.components.factories.processor import ProcessorFactory
from runtime.core.components.factories.reader import StreamReaderFactory
from runtime.core.components.factories.writer import StreamWriterFactory
from runtime.core.components.pipeline import PipelineRunner
from runtime.core.components.sink import Sink
from runtime.core.components.source import Source


class ComponentFactory(ABC):
    @abstractmethod
    def create_source(self, in_queue: Queue, reader_conf: Any) -> Source:
        ...

    @abstractmethod
    def create_pipeline(self, in_queue: Queue, broadcaster: FrameBroadcaster,
                        pipeline_config: Any) -> PipelineRunner:
        ...

    @abstractmethod
    def create_sink(self, broadcaster: FrameBroadcaster, writer_conf: Any) -> Sink:
        ...


class DefaultComponentFactory(ComponentFactory):

    def create_source(self, in_queue: Queue, reader_conf: Any) -> Source:
        return Source(in_queue, StreamReaderFactory.create(reader_conf))

    def create_pipeline(self, in_queue: Queue, broadcaster: FrameBroadcaster,
                        pipeline_config: Any) -> PipelineRunner:
        return PipelineRunner(in_queue, broadcaster, ProcessorFactory.create(pipeline_config))

    def create_sink(self, broadcaster: FrameBroadcaster, writer_conf: Any) -> Sink:
        return Sink(broadcaster, StreamWriterFactory.create(writer_conf))
