#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
import time
from queue import Empty, Queue, Full
from threading import Lock
from typing import List

from backend.app.runtime.core.base import Processor, StreamReader, StreamWriter, JobComponent

logger = logging.getLogger(__name__)


class Source(JobComponent):
    """Reads from a StreamReader and puts the data into the provided queue."""

    def __init__(self, in_queue: Queue, stream_reader: StreamReader):
        super().__init__()
        self._reader = stream_reader
        self._in_queue = in_queue

    def run(self) -> None:

        logger.debug("Starting a source loop")
        with self._reader:
            while not self._stop_event.is_set():
                try:
                    data = self._reader.read()
                    if data is None:
                        time.sleep(0.01)
                        continue
                    try:
                        self._in_queue.put_nowait(data)
                    except Full:
                        try:
                            self._in_queue.get_nowait()
                            self._in_queue.put_nowait(data)
                        except Full:
                            logger.error("Input queue still full after dropping. Skipping current data.")
                            pass
                except Exception as e:
                    logger.error(f"Error reading from stream: {e}.")
                    time.sleep(0.1)
        logger.debug("Stopping the source loop")


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


class FrameBroadcaster:
    """
    A thread-safe class to broadcast frames to multiple consumers.

    It manages a queue for each registered consumer. If a consumer's
    queue is full the oldest frame is dropped to make space for the new one.
    """

    def __init__(self):
        self.queues: List[Queue] = []
        self._lock = Lock()

    def register(self) -> Queue:
        """Register a new consumer and return its personal queue."""
        with self._lock:
            queue = Queue(maxsize=2)
            self.queues.append(queue)
            logging.info(f"Registered new consumer. Total consumers: {len(self.queues)}")
            return queue

    def unregister(self, queue: Queue):
        """Unregister a consumer by its queue."""
        with self._lock:
            try:
                self.queues.remove(queue)
                logging.info(f"Unregistered consumer. Total consumers: {len(self.queues)}")
            except ValueError:
                # if a client twice.
                pass

    def broadcast(self, frame):
        """Broadcast a frame to all registered consumers."""
        with self._lock:
            for queue in self.queues:
                try:
                    queue.put_nowait(frame)
                except Full:
                    logging.warning("Consumer queue is full. Dropping oldest frame.")
                    try:
                        queue.get_nowait()
                        queue.put_nowait(frame)
                    except Full:
                        pass


class PipelineRunner(JobComponent):
    """A component that delegates processing logic to a processor."""

    def __init__(self, in_queue: Queue, broadcaster: FrameBroadcaster, processor: Processor):

        super().__init__()
        self._processor = processor
        self._in_queue = in_queue
        self._broadcaster = broadcaster

    def run(self) -> None:
        logger.debug("Starting a pipeline runner loop")
        while not self._stop_event.is_set():
            try:
                data = self._in_queue.get(timeout=0.1)
                processed_data = self._processor.process(data)
                if processed_data is not None:
                    self._broadcaster.broadcast(processed_data)

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in pipeline runner loop: {e}")

        logger.debug("Stopping the pipeline runner loop")
