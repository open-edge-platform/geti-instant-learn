import threading
from queue import Queue
from typing import List, Dict

from backend.app.pipeline.core.base import Processor, PipelineComponent
from backend.app.pipeline.core.components import Source, SequenceProcessor, Sink
from backend.app.pipeline.core.factories import StreamReaderFactory, StreamWriterFactory
from types import ConfigDict


class Pipeline:
    """
    Orchestrates the pipeline lifecycle, including queue and thread management.
    """

    def __init__(self,
                 source_config: ConfigDict,
                 processors: List[Processor],
                 sink_config: ConfigDict):

        self._processors_config = processors
        self._in_queue = Queue(maxsize=5)
        self._out_queue = Queue(maxsize=5)

        source_reader = StreamReaderFactory.create(source_config)
        sink_writer = StreamWriterFactory.create(sink_config)

        source = Source(source_reader, self._in_queue)
        processor = SequenceProcessor(self._processors_config, self._in_queue, self._out_queue)
        sink = Sink(sink_writer, self._out_queue)

        self._components: Dict[str, PipelineComponent] = {
            c.name: c for c in [source, processor, sink]
        }
        self._threads: Dict[str, threading.Thread] = {}

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
