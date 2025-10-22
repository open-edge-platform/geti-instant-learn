#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
import time
from queue import Full, Queue

from core.components.base import PipelineComponent, StreamReader
from core.components.broadcaster import FrameBroadcaster
from core.components.schemas.processor import InputData

logger = logging.getLogger(__name__)


class Source(PipelineComponent):
    """Reads from a StreamReader and puts raw frames into the provided inbound queue and broadcasts to consumers."""

    def __init__(
        self,
        in_queue: Queue,
        stream_reader: StreamReader,
        inbound_broadcaster: FrameBroadcaster[InputData] | None = None,
    ):
        super().__init__()
        self._reader = stream_reader
        self._in_queue = in_queue
        self._inbound_broadcaster = inbound_broadcaster

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

                    if self._inbound_broadcaster:
                        self._inbound_broadcaster.broadcast(data)  # broadcast to inbound consumers (e.g. frame capture)

                    try:
                        self._in_queue.put_nowait(data)  # send to processor queue (with backpressure handling)
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
