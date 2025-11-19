# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from queue import Empty

from runtime.core.components.base import PipelineComponent, StreamWriter
from runtime.core.components.broadcaster import FrameBroadcaster

logger = logging.getLogger(__name__)


class Sink(PipelineComponent):
    """Gets data from a queue and writes it using a StreamWriter."""

    def __init__(self, broadcaster: FrameBroadcaster, stream_writer: StreamWriter):
        super().__init__()
        self._writer = stream_writer
        self.broadcaster = broadcaster
        logger.debug("Sink registering to OutboundBroadcaster for processed frames")
        self._out_queue = broadcaster.register()

    def run(self) -> None:
        logger.debug("Starting a sink loop")
        with self._writer:
            while not self._stop_event.is_set():
                try:
                    data = self._out_queue.get(timeout=0.1)
                    self._writer.write(data)
                except Empty:
                    continue
            logger.debug("Stopping the sink loop")

    def _stop(self) -> None:
        self.broadcaster.unregister(self._out_queue)
