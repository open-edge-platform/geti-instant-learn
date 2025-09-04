#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
import time
from multiprocessing import Queue
from queue import Empty

from backend.app.runtime.core.base import Processor, StreamReader, StreamWriter, JobComponent

logger = logging.getLogger(__name__)


class Source(JobComponent):
    """Reads from a StreamReader and puts the data into the provided queue."""

    def __init__(self, in_queue: Queue, stream_reader: StreamReader):
        super().__init__()
        self._reader = stream_reader
        self._in_queue = in_queue

    def run(self) -> None:

        logger.debug(f"Starting a source loop")
        with self._reader:
            while not self._stop_event.is_set():
                data = self._reader.read()
                if data is None:
                    time.sleep(0.1)
                    continue
                self._in_queue.put(data)

            logger.debug(f"Stopping the source loop")


class Sink(JobComponent):
    """Gets data from a queue and writes it using a StreamWriter."""

    def __init__(self, out_queue: Queue, stream_writer: StreamWriter):
        super().__init__()
        self._writer = stream_writer
        self._out_queue = out_queue

    def run(self) -> None:
        logger.debug(f"Starting a sink loop")
        with self._writer:
            while not self._stop_event.is_set():
                try:
                    data = self._out_queue.get(timeout=0.1)
                    self._writer.write(data)
                except Empty:
                    continue
            logger.debug(f"Stopping the sink loop")


class PipelineRunner(JobComponent):
    """A component that delegates processing logic to a processor."""

    def __init__(self, in_queue: Queue, out_queue: Queue, processor: Processor):

        super().__init__()
        self._processor = processor
        self._in_queue = in_queue
        self._out_queue = out_queue

    def run(self) -> None:
        logger.debug(f"Starting a pipeline runner loop")
        while not self._stop_event.is_set():
            try:
                data = self._in_queue.get(timeout=0.1)
                processed_data = self._processor.process(data)
                self._out_queue.put(processed_data)
            except Empty:
                continue
        logger.debug(f"Stopping the pipeline runner loop")
