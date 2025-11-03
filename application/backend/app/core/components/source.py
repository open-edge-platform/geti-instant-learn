import logging
import threading
import time

from core.components.base import PipelineComponent, StreamReader
from core.components.broadcaster import FrameBroadcaster
from core.components.schemas.processor import InputData
from core.components.schemas.reader import FrameListResponse

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
        self._pause_condition = threading.Condition()

    def run(self) -> None:
        logger.debug("Starting a source loop")
        with self._reader:
            self._reader.connect()
            while not self._stop_event.is_set():
                # Wait while paused using condition variable
                with self._pause_condition:
                    while self._pause_event.is_set() and not self._stop_event.is_set():
                        self._pause_condition.wait()

                if self._stop_event.is_set():
                    break

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

    def seek(self, index: int) -> InputData | None:
        """
        Seek to a specific frame index.
        Delegates to reader.seek() and returns the frame data.
        """
        return self._reader.seek(index)

    def index(self) -> int:
        """
        Get current frame position.
        Delegates to reader.index().
        """
        return self._reader.index()

    def list_frames(self, page: int = 1, page_size: int = 100) -> FrameListResponse:
        """
        Get paginated list of all frames.
        Delegates to reader.list_frames().
        """
        return self._reader.list_frames(page, page_size)

    def stop(self) -> None:
        """
        Stop source flow.
        Sets stop event and notifies pause condition to wake up waiting threads.
        """
        self._stop_event.set()
        # Wake up thread if it's waiting on pause condition
        with self._pause_condition:
            self._pause_condition.notify()
        logger.debug("Source stopped")
