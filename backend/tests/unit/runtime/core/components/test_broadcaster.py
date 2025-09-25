from queue import Queue

from runtime.core.components.broadcaster import FrameBroadcaster


class TestFrameBroadcaster:

    def setup_method(self, method):
        self.broadcaster = FrameBroadcaster()

    def test_register_adds_new_queue(self):
        assert len(self.broadcaster.queues) == 0

        q1 = self.broadcaster.register()
        q2 = self.broadcaster.register()

        assert len(self.broadcaster.queues) == 2
        assert q1 in self.broadcaster.queues
        assert q2 in self.broadcaster.queues
        assert q1 is not q2

    def test_unregister_removes_queue(self):
        q1 = self.broadcaster.register()
        q2 = self.broadcaster.register()
        assert len(self.broadcaster.queues) == 2

        self.broadcaster.unregister(q1)

        assert len(self.broadcaster.queues) == 1
        assert q1 not in self.broadcaster.queues
        assert q2 in self.broadcaster.queues

    def test_unregister_is_safe_for_non_existent_queue(self):
        q1 = self.broadcaster.register()
        a_non_existent_queue = Queue()

        self.broadcaster.unregister(a_non_existent_queue)
        assert len(self.broadcaster.queues) == 1

        self.broadcaster.unregister(q1)
        assert len(self.broadcaster.queues) == 0

        self.broadcaster.unregister(q1)
        assert len(self.broadcaster.queues) == 0

    def test_broadcast_sends_to_all_consumers(self):
        q1 = self.broadcaster.register()
        q2 = self.broadcaster.register()
        frame = "test_frame"

        self.broadcaster.broadcast(frame)

        assert q1.get_nowait() == frame
        assert q2.get_nowait() == frame

    def test_broadcast_drops_oldest_frame_for_slow_consumer(self):
        fast_consumer_q = self.broadcaster.register()
        slow_consumer_q = self.broadcaster.register()

        # Simulate a slow consumer by filling its queue to capacity.
        slow_consumer_q.put_nowait("frame0")
        slow_consumer_q.put_nowait("frame1")
        assert slow_consumer_q.full()

        self.broadcaster.broadcast("frame2")

        assert fast_consumer_q.qsize() == 1
        assert fast_consumer_q.get_nowait() == "frame2"

        # The slow consumer's queue has dropped the oldest frame ("frame0")
        assert slow_consumer_q.full()
        assert slow_consumer_q.get_nowait() == "frame1"
        assert slow_consumer_q.get_nowait() == "frame2"
