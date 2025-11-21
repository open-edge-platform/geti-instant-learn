# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from queue import Empty

from domain.services.schemas.processor import OutputData
from runtime.core.components.base import PipelineComponent, StreamWriter
from runtime.core.components.broadcaster import FrameBroadcaster

logger = logging.getLogger(__name__)


class Sink(PipelineComponent):
    """Gets data from a queue and writes it using a StreamWriter."""

    def __init__(self, stream_writer: StreamWriter):
        super().__init__()
        self._writer = stream_writer
        self._initialized = False

    def setup(self, outbound_broadcaster: FrameBroadcaster[OutputData]) -> None:
        self._out_queue = outbound_broadcaster.register()
        self._outbound_broadcaster = outbound_broadcaster
        self._initialized = True

    def run(self) -> None:
        if not self._initialized:
            raise RuntimeError("The sink should be initialized before being used")
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
        self._outbound_broadcaster.unregister(self._out_queue)
