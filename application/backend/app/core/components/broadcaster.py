# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from queue import Full, Queue
from threading import Lock

logger = logging.getLogger(__name__)


class FrameBroadcaster[T]:
    """
    A thread-safe class to broadcast frames to multiple consumers.

    It manages a queue for each registered consumer. If a consumer's
    queue is full the oldest frame is dropped to make space for the new one.

    The live nature of WebRTC streams requires consumers to be registered and unregistered dynamically as they connect
    and disconnect. If we were to share a single queue for all consumers, they would compete for frames, effectively
    stealing them from each other. This broadcaster ensures every consumer gets its own queue.
    """

    def __init__(self):
        self.queues: list[Queue[T]] = []
        self._lock = Lock()

    def register(self) -> Queue:
        """Register a new consumer and return its personal queue."""
        with self._lock:
            queue = Queue[T](maxsize=2)
            self.queues.append(queue)
            logging.info("Registered new consumer. Total consumers: %", len(self.queues))
            return queue

    def unregister(self, queue: Queue) -> None:
        """Unregister a consumer by its queue."""
        with self._lock:
            try:
                self.queues.remove(queue)
                logging.info("Unregistered consumer. Total consumers:%", len(self.queues))
            except ValueError:
                # if a client unregisters twice.
                pass

    def broadcast(self, frame: T) -> None:
        """Broadcast a frame to all registered consumers."""
        with self._lock:
            for queue in self.queues:
                try:
                    queue.put_nowait(frame)
                except Full:
                    logging.warning("Consumer queue is full. Dropping oldest frame.")
                    try:
                        queue.get_nowait()
                        queue.put_nowait(frame)
                    except Full:
                        pass
