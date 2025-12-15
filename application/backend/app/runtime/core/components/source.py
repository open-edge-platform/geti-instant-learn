#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
import time

from domain.services.schemas.processor import InputData
from domain.services.schemas.reader import FrameListResponse
from runtime.core.components.base import PipelineComponent, StreamReader
from runtime.core.components.broadcaster import FrameBroadcaster

logger = logging.getLogger(__name__)


class Source(PipelineComponent):
    """Reads from a StreamReader and broadcasts raw frames to registered consumers."""

    def __init__(
        self,
        stream_reader: StreamReader,
    ):
        super().__init__()
        self._reader = stream_reader
        self._initialized = False

    def setup(self, inbound_broadcaster: FrameBroadcaster[InputData]) -> None:
        self._inbound_broadcaster = inbound_broadcaster
        self._initialized = True

    def run(self) -> None:
        if not self._initialized:
            raise RuntimeError("The source should be initialized before being used")

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

    def seek(self, index: int) -> None:
        """
        Seek to a specific frame index.
        Delegates to reader.seek().
        """
        self._reader.seek(index)

    def index(self) -> int:
        """
        Get current frame position.
        Delegates to reader.index().
        """
        return self._reader.index()

    def list_frames(self, offset: int = 0, limit: int = 30) -> FrameListResponse:
        """
        Get paginated list of all frames.
        Delegates to reader.list_frames().
        """
        return self._reader.list_frames(offset=offset, limit=limit)
