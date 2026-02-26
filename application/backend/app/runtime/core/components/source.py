#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
import time
from threading import Condition

from domain.services.schemas.processor import InputData
from domain.services.schemas.reader import FrameListResponse
from runtime.core.components.base import PipelineComponent, StreamReader
from runtime.core.components.broadcaster import FrameBroadcaster

logger = logging.getLogger(__name__)


class Source(PipelineComponent):
    """Reads from a StreamReader and broadcasts raw frames to registered consumers.

    Supports two reader modes:
    - **Auto-advancing**: Video cameras, video files, etc. Frames are read continuously
      in a loop without user involvement.
    - **Manual**: Image folders, etc. Requires user to explicitly request the next frame
      (except the first frame, which is shown automatically).

    The flow control mode is determined by the reader's `requires_manual_control` property.
    """

    def __init__(
        self,
        stream_reader: StreamReader,
    ):
        super().__init__()
        self._reader = stream_reader
        self._initialized = False
        self._inbound_broadcaster: FrameBroadcaster[InputData] | None = None
        self._manual_mode = self._reader.requires_manual_control
        self._next_frame_condition = Condition()
        self._next_frame_requested = True

    def setup(self, inbound_broadcaster: FrameBroadcaster[InputData]) -> None:
        self._inbound_broadcaster = inbound_broadcaster
        self._initialized = True

    def run(self) -> None:
        if not self._initialized or self._inbound_broadcaster is None:
            raise RuntimeError("The source should be initialized before being used")

        self._reader.connect()

        logger.debug(f"Starting a source {self._reader.__class__.__name__} loop")
        while not self._stop_event.is_set():
            if self._manual_mode:
                with self._next_frame_condition:
                    while not self._next_frame_requested and not self._stop_event.is_set():
                        self._next_frame_condition.wait()

                    if self._stop_event.is_set():
                        break

                    self._next_frame_requested = False

            try:
                data = self._reader.read()
                if data is None:
                    time.sleep(0.01)
                    continue

                self._inbound_broadcaster.broadcast(data)

            except Exception as e:
                logger.exception(f"Error reading from stream: {e}.")
                time.sleep(0.1)
        logger.debug(f"Stopping the source {self._reader.__class__.__name__} loop")
        # TODO: To investigate why reader.close() is fixing issue when switching cameras
        self._reader.close()

    def _stop(self) -> None:
        """Clean up resources when component is stopped."""
        with self._next_frame_condition:
            self._next_frame_condition.notify_all()
        try:
            self._reader.close()
        except Exception as e:
            logger.exception(f"Error closing reader: {e}")

    def seek(self, index: int) -> None:
        """
        Seek to a specific frame index.
        Delegates to reader.seek().
        """
        self._reader.seek(index)
        if self._manual_mode:
            with self._next_frame_condition:
                self._next_frame_requested = True
                self._next_frame_condition.notify()

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
