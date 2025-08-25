import time
from queue import Queue, Empty
from typing import List

from backend.app.pipeline.core.base import PipelineComponent, Processor, StreamReader, StreamWriter
from types import IN, OUT


class Source(PipelineComponent):
    """Reads from a StreamReader and puts the data into the provided queue."""

    def __init__(self, stream_reader: StreamReader, in_queue: Queue):
        super().__init__(name="Source")
        self._reader = stream_reader
        self._in_queue = in_queue

    def _main_loop(self) -> None:
        with self._reader:
            while not self._stop_event.is_set():
                data = self._reader.read()
                if data is None:
                    time.sleep(0.1)
                    continue
                self._in_queue.put(data)
        self._in_queue.put(None)


class Sink(PipelineComponent):
    """Gets data from a queue and writes it using a StreamWriter."""

    def __init__(self, stream_writer: StreamWriter, in_queue: Queue):
        super().__init__(name="Sink")
        self._writer = stream_writer
        self._in_queue = in_queue

    def _main_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                data = self._in_queue.get(timeout=0.1)
                if data is None:
                    break
                self._writer.write(data)
            except Empty:
                continue


class SequenceProcessor(PipelineComponent, Processor[IN, OUT]):
    """A component and a composite processor impl that runs a sequence of processors."""

    def __init__(self, processors: List[Processor], in_queue: Queue, out_queue: Queue):
        super().__init__(name="Processor")
        self._processors = processors
        self._in_queue = in_queue
        self._out_queue = out_queue

    def _main_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                data = self._in_queue.get(timeout=0.1)
                if data is None:
                    self._in_queue.put(None)
                    break
                processed_data = self.process(data)
                self._out_queue.put(processed_data)
            except Empty:
                continue
        self._out_queue.put(None)

    def process(self, input_data: IN) -> OUT:
        result = input_data
        for processor in self._processors:
            result = processor.process(result)
        return result
