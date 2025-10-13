#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
import time
from queue import Full, Queue

from core.components.base import PipelineComponent, StreamReader

logger = logging.getLogger(__name__)


class Source(PipelineComponent):
    """Reads from a StreamReader and puts raw frames into the provided inbound queue."""

    def __init__(self, in_queue: Queue, stream_reader: StreamReader):
        super().__init__()
        self._reader = stream_reader
        self._in_queue = in_queue

    def run(self) -> None:
        logger.debug("Starting a source loop")
        with self._reader:
            self._reader.connect()
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
                except Exception as e:
                    logger.error(f"Error reading from stream: {e}.")
                    time.sleep(0.1)
        logger.debug("Stopping the source loop")
