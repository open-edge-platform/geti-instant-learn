import multiprocessing
from unittest.mock import MagicMock

from backend.app.runtime.core.base import StreamReader
from backend.app.runtime.core.components import Source


class TestSource():

    def setup_method(self, method):
        self.manager = multiprocessing.Manager()
        self.in_queue = self.manager.Queue()
        self.mock_stream_reader = MagicMock(spec=StreamReader)
        self.mock_stream_reader.__enter__.return_value = self.mock_stream_reader
        self.source = Source(self.in_queue, self.mock_stream_reader)

    def teardown_method(self, method):
        self.source.stop()
        self.manager.shutdown()

    def test_run_reads_and_queues_data(self):
        original_test_data = ["frame1", "frame2", "frame3"]
        iterator = iter(original_test_data)

        def read_and_then_stop(*args, **kwargs):
            try:
                return next(iterator)
            except StopIteration:
                self.source.stop()
                return None

        self.mock_stream_reader.read.side_effect = read_and_then_stop
        self.source.run()

        self.mock_stream_reader.__enter__.assert_called_once()
        self.mock_stream_reader.__exit__.assert_called_once()

        assert self.in_queue.qsize() == len(original_test_data)
        results = []
        while not self.in_queue.empty():
            results.append(self.in_queue.get())
        assert results == original_test_data
