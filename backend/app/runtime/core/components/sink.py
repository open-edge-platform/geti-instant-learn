# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from queue import Queue, Empty

from runtime.core.components.base import JobComponent
from runtime.core.components.base import StreamWriter

logger = logging.getLogger(__name__)


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
