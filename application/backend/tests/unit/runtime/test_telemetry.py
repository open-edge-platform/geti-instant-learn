# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

from domain.services.schemas.frame_trace import FrameTrace
from runtime.telemetry import (
    ProcessorStats,
    SourceStats,
    TelemetryComponent,
    WebRtcFrameKind,
    WebRtcRecvStats,
    log_trace,
    trace_span,
)


def test_trace_span_records_component_timing():
    trace = FrameTrace(frame_id="frame-id")

    with patch.object(FrameTrace, "_now_ms", side_effect=[10.0, 15.0]):
        with trace_span(trace, TelemetryComponent.WEBRTC):
            pass

    assert len(trace.spans) == 1
    assert trace.spans[0].component == "webrtc"
    assert trace.spans[0].duration_ms == 5.0


def test_trace_span_accepts_missing_trace():
    with trace_span(None, TelemetryComponent.WEBRTC):
        pass


def test_log_trace_logs_when_trace_is_available():
    logger = Mock()
    trace = Mock()
    trace.format_log.return_value = "trace log"

    log_trace(trace, logger)

    logger.debug.assert_called_once_with("trace log")


def test_log_trace_accepts_missing_trace():
    logger = Mock()

    log_trace(None, logger)

    logger.debug.assert_not_called()


def test_source_stats_logs_interval_and_resets():
    logger = Mock()

    with patch("runtime.telemetry.time.perf_counter", side_effect=[0.0, 1.0]):
        stats = SourceStats(interval_secs=1, logger=logger, reader_name="Reader", manual_mode=True)
        stats.record(frame_produced=True, read_wall_time_s=0.004, read_cpu_time_s=0.002)

    logger.info.assert_called_once()
    assert logger.info.call_args.args[1:] == ("Reader", True, 1.0, 1, 1.0, 0, 4.0, 2.0)
    assert stats.frame_count == 0
    assert stats.empty_read_count == 0


def test_processor_stats_logs_interval_and_resets():
    logger = Mock()

    with patch("runtime.telemetry.time.perf_counter", side_effect=[0.0, 1.0]):
        stats = ProcessorStats(interval_secs=1, logger=logger)
        stats.record_input_frame()
        stats.record_skipped_frame()
        stats.record_batch(batch_size=2, wall_time_s=0.005, cpu_time_s=0.003)

    logger.info.assert_called_once()
    assert logger.info.call_args.args[1:] == (1.0, 1, 1.0, 1, 1, 2, 2.0, 0, 0, 5.0, 3.0)
    assert stats.batch_count == 0
    assert stats.output_frame_count == 0


def test_web_rtc_recv_stats_logs_interval_and_resets():
    logger = Mock()

    with patch("runtime.telemetry.time.perf_counter", side_effect=[0.0, 1.0]):
        stats = WebRtcRecvStats(interval_secs=1, logger=logger)
        stats.record(frame_kind=WebRtcFrameKind.CACHED, wall_time_s=0.002, cpu_time_s=0.001)

    logger.info.assert_called_once()
    assert logger.info.call_args.args[1:] == (1.0, 1, 1.0, 0, 1, 0, 0, 2.0, 1.0)
    assert stats.recv_count == 0
    assert stats.cached_frame_count == 0
