#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
import time
from threading import Condition

from domain.dispatcher import ComponentType
from domain.services.schemas.frame_trace import FrameTrace
from domain.services.schemas.processor import ErrorData, InputData
from domain.services.schemas.reader import FrameListResponse
from runtime.core.components.base import PipelineComponent, StreamReader
from runtime.core.components.broadcaster import FrameBroadcaster
from runtime.telemetry import SourceStats, TelemetryComponent, trace_span
from settings import get_settings

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
        self._inbound_broadcaster: FrameBroadcaster[InputData | ErrorData] | None = None
        self._manual_mode = self._reader.requires_manual_control
        self._next_frame_condition = Condition()
        self._next_frame_requested = True
        settings = get_settings()
        self._stats_enabled = settings.cpu_monitoring_enabled
        self._stats = SourceStats(
            interval_secs=settings.cpu_monitoring_interval_secs,
            logger=logger,
            reader_name=self._reader.__class__.__name__,
            manual_mode=self._manual_mode,
        )

    def setup(self, inbound_broadcaster: FrameBroadcaster[InputData | ErrorData]) -> None:
        self._inbound_broadcaster = inbound_broadcaster
        self._initialized = True

    def run(self) -> None:
        if not self._initialized or self._inbound_broadcaster is None:
            raise RuntimeError("The source should be initialized before being used")

        try:
            self._reader.connect()
        except Exception as e:
            logger.exception(f"Failed to connect reader: {e}")
            self._inbound_broadcaster.broadcast(ErrorData(message=str(e), component=ComponentType.SOURCE))
            return

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
                trace = FrameTrace.create()

                read_wall_started_at_s = time.perf_counter()
                read_cpu_started_at_s = time.thread_time()
                with trace_span(trace, TelemetryComponent.SOURCE):
                    data = self._reader.read()
                read_wall_time_s = time.perf_counter() - read_wall_started_at_s
                read_cpu_time_s = time.thread_time() - read_cpu_started_at_s
                if data is None:
                    self._record_source_stats(
                        frame_produced=False,
                        read_wall_time_s=read_wall_time_s,
                        read_cpu_time_s=read_cpu_time_s,
                    )
                    time.sleep(0.01)
                    continue

                data.trace = trace

                self._record_source_stats(
                    frame_produced=True,
                    read_wall_time_s=read_wall_time_s,
                    read_cpu_time_s=read_cpu_time_s,
                )
                self._inbound_broadcaster.broadcast(data)

            except Exception as e:
                logger.exception(f"Error reading from stream: {e}.")
                self._inbound_broadcaster.broadcast(ErrorData(message=str(e), component=ComponentType.SOURCE))

        logger.debug(f"Stopping the source {self._reader.__class__.__name__} loop")
        # TODO: To investigate why reader.close() is fixing issue when switching cameras
        self._reader.close()

    def _record_source_stats(self, frame_produced: bool, read_wall_time_s: float, read_cpu_time_s: float) -> None:
        if not self._stats_enabled:
            return
        self._stats.record(
            frame_produced=frame_produced,
            read_wall_time_s=read_wall_time_s,
            read_cpu_time_s=read_cpu_time_s,
        )

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
