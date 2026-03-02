# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from queue import Empty, Full, Queue
from threading import Lock

logger = logging.getLogger(__name__)


class FrameSlot[T]:
    """A shared container holding the latest frame for external consumers."""

    def __init__(self) -> None:
        self._frame: T | None = None

    @property
    def latest(self) -> T | None:
        """The most recently published frame, or None if nothing has been published."""
        return self._frame

    def update(self, frame: T) -> None:
        """Publish a new frame, replacing any previously held value."""
        self._frame = frame

    def clear(self) -> None:
        """Discard the held frame."""
        self._frame = None


class FrameBroadcaster[T]:
    """
    A thread-safe class to broadcast frames to multiple consumers.

    It manages a named queue for each registered consumer. If a consumer's
    queue is full the oldest frame is dropped to make space for the new one.

    A FrameSlot is maintained alongside the queues so that external consumers
    (e.g. WebRTC streams) can poll the latest frame without registering a queue.
    """

    def __init__(self, name: str = "unnamed") -> None:
        self.name = name
        self._consumers: dict[str, Queue[T]] = {}
        self._lock = Lock()
        self._slot: FrameSlot[T] = FrameSlot[T]()

    @property
    def slot(self) -> FrameSlot[T]:
        """Shared slot that always holds the latest broadcasted frame."""
        return self._slot

    @property
    def latest_frame(self) -> T | None:
        """Get the most recently broadcasted frame."""
        return self._slot.latest

    @property
    def consumer_count(self) -> int:
        """Number of registered consumers."""
        return len(self._consumers)

    def register(self, consumer_name: str) -> Queue[T]:
        """Register a new consumer and return its personal queue.

        If a frame has already been broadcast, the latest frame is immediately
        added to the new consumer's queue so they don't miss the current state.
        """
        with self._lock:
            queue: Queue[T] = Queue(maxsize=5)
            self._consumers[consumer_name] = queue

            # Send the latest frame to new consumer if available
            if self._slot.latest is not None:
                try:
                    queue.put_nowait(self._slot.latest)
                except Full:
                    logging.warning("Could not send latest frame to new consumer - queue full")

            logging.info(
                "%s: registered consumer '%s'. Total consumers: %d", self.name, consumer_name, len(self._consumers)
            )
            return queue

    def unregister(self, consumer_name: str) -> None:
        """Unregister a consumer by name."""
        with self._lock:
            if self._consumers.pop(consumer_name, None) is not None:
                logging.info(
                    "%s: unregistered consumer '%s'. Total consumers: %d",
                    self.name,
                    consumer_name,
                    len(self._consumers),
                )

    def broadcast(self, frame: T) -> None:
        """Broadcast frame to all registered queues and update the shared slot."""
        self._slot.update(frame)
        with self._lock:
            for consumer_name, queue in self._consumers.items():
                try:
                    queue.put_nowait(frame)
                except Full:
                    self._handle_full_queue(consumer_name, queue, frame)
                except Exception:
                    logger.exception("Error broadcasting to queue")
                logger.debug("%s/%s depth: %d/%d", self.name, consumer_name, queue.qsize(), queue.maxsize)

    def clear(self) -> None:
        """
        Drop all queued frames for all consumers.
        Keeps consumer queues registered, but drains them so no stale frames are delivered
        after a component swap (e.g., changing the source).
        """
        with self._lock:
            for consumer_name, q in self._consumers.items():
                while True:
                    try:
                        q.get_nowait()
                    except Empty:
                        logger.debug("Drained queued frames for consumer '%s'", consumer_name)
                        break
            self._slot.clear()

    def _handle_full_queue(self, consumer_name: str, queue: Queue[T], frame: T) -> None:
        """Handle a full queue by dropping the oldest frame and adding the new one."""
        logger.warning(
            "%s/%s full (%d/%d), dropping oldest frame", self.name, consumer_name, queue.qsize(), queue.maxsize
        )
        try:
            queue.get_nowait()
        except Empty:
            pass

        try:
            queue.put_nowait(frame)
        except Full:
            logger.warning("Queue still full after clearing, skipping frame")
        except Exception:
            logger.exception("Error replacing frame in full queue")
