from queue import Queue
from typing import Any

import pytest

from runtime.core.components.broadcaster import FrameBroadcaster


@pytest.fixture
def broadcaster() -> FrameBroadcaster[Any]:
    return FrameBroadcaster()


class TestFrameBroadcaster:
    def test_register_adds_new_queue(self, broadcaster):
        assert len(broadcaster.queues) == 0

        q1 = broadcaster.register()
        q2 = broadcaster.register()

        assert len(broadcaster.queues) == 2
        assert q1 in broadcaster.queues
        assert q2 in broadcaster.queues
        assert q1 is not q2

    def test_unregister_removes_queue(self, broadcaster):
        q1 = broadcaster.register()
        q2 = broadcaster.register()
        assert len(broadcaster.queues) == 2

        broadcaster.unregister(q1)

        assert len(broadcaster.queues) == 1
        assert q1 not in broadcaster.queues
        assert q2 in broadcaster.queues

    def test_unregister_is_safe_for_non_existent_queue(self, broadcaster):
        q1 = broadcaster.register()
        a_non_existent_queue = Queue()

        broadcaster.unregister(a_non_existent_queue)
        assert len(broadcaster.queues) == 1

        broadcaster.unregister(q1)
        assert len(broadcaster.queues) == 0

        broadcaster.unregister(q1)
        assert len(broadcaster.queues) == 0

    def test_broadcast_sends_to_all_consumers(self, broadcaster):
        q1 = broadcaster.register()
        q2 = broadcaster.register()
        frame = "test_frame"

        broadcaster.broadcast(frame)

        assert q1.get_nowait() == frame
        assert q2.get_nowait() == frame

    def test_broadcast_drops_oldest_frame_for_slow_consumer(self, broadcaster):
        fast_consumer_q = broadcaster.register()
        slow_consumer_q = broadcaster.register()

        # Simulate a slow consumer by filling its queue to capacity (maxsize=5).
        slow_consumer_q.put_nowait("frame0")
        slow_consumer_q.put_nowait("frame1")
        slow_consumer_q.put_nowait("frame2")
        slow_consumer_q.put_nowait("frame3")
        slow_consumer_q.put_nowait("frame4")
        assert slow_consumer_q.full()

        broadcaster.broadcast("frame5")

        assert fast_consumer_q.qsize() == 1
        assert fast_consumer_q.get_nowait() == "frame5"

        # The slow consumer's queue has dropped the oldest frame ("frame0")
        assert slow_consumer_q.full()
        assert slow_consumer_q.get_nowait() == "frame1"
        assert slow_consumer_q.get_nowait() == "frame2"
        assert slow_consumer_q.get_nowait() == "frame3"
        assert slow_consumer_q.get_nowait() == "frame4"
        assert slow_consumer_q.get_nowait() == "frame5"

    def test_register_receives_latest_frame_when_available(self, broadcaster):
        q1 = broadcaster.register()
        assert q1.empty()

        frame1 = "test_frame_1"
        broadcaster.broadcast(frame1)
        assert q1.get_nowait() == frame1

        q2 = broadcaster.register()
        assert not q2.empty()
        assert q2.get_nowait() == frame1

        frame2 = "test_frame_2"
        broadcaster.broadcast(frame2)

        q3 = broadcaster.register()
        assert not q3.empty()
        assert q3.get_nowait() == frame2

    def test_register_without_broadcast_has_empty_queue(self, broadcaster):
        q = broadcaster.register()
        assert q.empty()
        assert broadcaster.latest_frame is None

    def test_latest_frame_property_updates(self, broadcaster):
        assert broadcaster.latest_frame is None

        frame1 = "frame_1"
        broadcaster.broadcast(frame1)
        assert broadcaster.latest_frame == frame1

        frame2 = "frame_2"
        broadcaster.broadcast(frame2)
        assert broadcaster.latest_frame == frame2
