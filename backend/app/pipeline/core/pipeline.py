#  Copyright (C) 2022-2025 Intel Corporation
#  LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import threading
from queue import Queue

from backend.app.pipeline.core.base import PipelineComponent, Processor
from backend.app.pipeline.core.components import SequenceProcessor, Sink, Source
from backend.app.pipeline.core.factories import StreamReaderFactory, StreamWriterFactory

from .types import ConfigDict


class Pipeline:
    """
    Orchestrates the pipeline lifecycle, including queue and thread management.
    """

    def __init__(self,
                 source_config: ConfigDict,
                 processors: list[Processor],
                 sink_config: ConfigDict):

        self._processors_config = processors
        self._in_queue = Queue(maxsize=5)
        self._out_queue = Queue(maxsize=5)

        source_reader = StreamReaderFactory.create(source_config)
        sink_writer = StreamWriterFactory.create(sink_config)

        source = Source(source_reader, self._in_queue)
        processor = SequenceProcessor(self._processors_config, self._in_queue, self._out_queue)
        sink = Sink(sink_writer, self._out_queue)

        self._components: dict[str, PipelineComponent] = {
            c.name: c for c in [source, processor, sink]
        }
        self._threads: dict[str, threading.Thread] = {}

    def start(self):
        for name, component in self._components.items():
            self._threads[name] = threading.Thread(target=component, name=name)
            self._threads[name].start()

    def stop(self):

        for component in self._components.values():
            component.stop()

        for thread in self._threads.values():
            if thread.is_alive():
                thread.join()
