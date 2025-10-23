#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
import time

from core.components.base import PipelineComponent, StreamReader
from core.components.broadcaster import FrameBroadcaster
from core.components.schemas.processor import InputData

logger = logging.getLogger(__name__)


class Source(PipelineComponent):
    """Reads from a StreamReader and broadcasts raw frames to registered consumers."""

    def __init__(
        self,
        stream_reader: StreamReader,
        inbound_broadcaster: FrameBroadcaster[InputData],
    ):
        super().__init__()
        self._reader = stream_reader
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

                    self._inbound_broadcaster.broadcast(data)

                except Exception as e:
                    logger.error(f"Error reading from stream: {e}.")
                    time.sleep(0.1)
        logger.debug("Stopping the source loop")
