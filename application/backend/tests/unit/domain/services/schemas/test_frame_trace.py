from unittest.mock import patch

from domain.services.schemas.frame_trace import ComponentSpan, FrameTrace


class TestComponentSpan:
    def test_duration_when_closed(self):
        span = ComponentSpan(start_ms=100.0, component="source", end_ms=150.0)
        assert span.duration_ms == 50.0

    def test_duration_when_open(self):
        span = ComponentSpan(start_ms=100.0, component="source")
        assert span.duration_ms is None

    def test_ordering_by_start_ms(self):
        early = ComponentSpan(start_ms=10.0, component="source")
        late = ComponentSpan(start_ms=20.0, component="processor")
        assert early < late


class TestFrameTrace:
    def test_record_start_and_end(self):
        trace = FrameTrace(frame_id="test123")
        trace.record_start("source")
        trace.record_end("source")

        assert len(trace.spans) == 1
        span = trace.spans[0]
        assert span.component == "source"
        assert span.end_ms is not None
        assert span.duration_ms >= 0

    def test_record_end_for_unknown_component_is_noop(self):
        trace = FrameTrace(frame_id="test123")
        trace.record_end("unknown")
        assert len(trace.spans) == 0

    def test_open_spans_cleared_after_record_end(self):
        trace = FrameTrace(frame_id="test123")
        trace.record_start("source")
        assert "source" in trace._open_spans
        trace.record_end("source")
        assert "source" not in trace._open_spans

    def test_spans_preserve_insertion_order(self):
        times = iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        trace = FrameTrace(frame_id="test123")
        with patch.object(FrameTrace, "_now_ms", side_effect=times):
            trace.record_start("source")
            trace.record_end("source")
            trace.record_start("processor")
            trace.record_end("processor")
            trace.record_start("webrtc")
            trace.record_end("webrtc")

        components = [s.component for s in trace.spans]
        assert components == ["source", "processor", "webrtc"]

    def test_format_log_full_pipeline(self):
        times = iter([100.0, 110.0, 120.0, 165.0, 170.0, 171.0])

        trace = FrameTrace(frame_id="abc123def456")
        with patch.object(FrameTrace, "_now_ms", side_effect=times):
            trace.record_start("source")
            trace.record_end("source")
            trace.record_start("processor")
            trace.record_end("processor")
            trace.record_start("webrtc")
            trace.record_end("webrtc")

        log = trace.format_log()
        assert log.startswith("[frame abc123def456]")
        assert "source: 10.00 ms" in log
        assert "processor: 45.00 ms" in log
        assert "webrtc: 1.00 ms" in log
        assert "wall: 71.00 ms" in log
        assert "total: 56.00 ms" in log

    def test_format_log_with_open_span(self):
        times = iter([100.0, 110.0, 120.0])

        trace = FrameTrace(frame_id="open_test")
        with patch.object(FrameTrace, "_now_ms", side_effect=times):
            trace.record_start("source")
            trace.record_end("source")
            trace.record_start("processor")

        log = trace.format_log()
        assert "source: 10.00 ms" in log
        assert "processor: open" in log
        # wall should not appear when last span is open
        assert "wall:" not in log

    def test_format_log_empty_trace(self):
        trace = FrameTrace(frame_id="empty")
        log = trace.format_log()
        assert log == "[frame empty] | total: 0.00 ms"
