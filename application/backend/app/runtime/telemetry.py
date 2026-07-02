# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from logging import Logger

from domain.services.schemas.frame_trace import FrameTrace


class TelemetryComponent(StrEnum):
    """Pipeline component names shared by trace spans and interval stats."""

    SOURCE = "source"
    PROCESSOR = "processor"
    SINK = "sink"
    WEBRTC = "webrtc"


class WebRtcFrameKind(StrEnum):
    """Frame categories emitted by the WebRTC track recv loop."""

    NEW = "new"
    CACHED = "cached"
    FALLBACK = "fallback"
    ERROR = "error"


@contextmanager
def trace_span(trace: FrameTrace | None, component: TelemetryComponent) -> Iterator[None]:
    """Record a component span when a per-frame trace is available."""
    if trace is None:
        yield
        return

    trace.record_start(component.value)
    try:
        yield
    finally:
        trace.record_end(component.value)


def log_trace(trace: FrameTrace | None, logger: Logger) -> None:
    """Log a per-frame trace when one is available."""
    if trace is not None:
        logger.debug(trace.format_log())


@dataclass(kw_only=True)
class SourceStats:
    """Interval stats for source reader activity."""

    interval_secs: int
    logger: Logger
    reader_name: str
    manual_mode: bool
    started_at_s: float = 0.0
    frame_count: int = 0
    empty_read_count: int = 0
    read_wall_time_s: float = 0.0
    read_cpu_time_s: float = 0.0

    def __post_init__(self) -> None:
        self.reset(time.perf_counter())

    def record(self, frame_produced: bool, read_wall_time_s: float, read_cpu_time_s: float) -> None:
        if frame_produced:
            self.frame_count += 1
        else:
            self.empty_read_count += 1
        self.read_wall_time_s += read_wall_time_s
        self.read_cpu_time_s += read_cpu_time_s

        now_s = time.perf_counter()
        elapsed_s = now_s - self.started_at_s
        if elapsed_s < self.interval_secs:
            return

        read_count = self.frame_count + self.empty_read_count
        frame_rate = self.frame_count / elapsed_s if elapsed_s > 0 else 0.0
        avg_read_wall_ms = (self.read_wall_time_s / read_count) * 1000 if read_count else 0.0
        avg_read_cpu_ms = (self.read_cpu_time_s / read_count) * 1000 if read_count else 0.0
        self.logger.info(
            "source_stats reader=%s manual_mode=%s interval_secs=%.1f frames=%d frame_rate=%.1f empty_reads=%d "
            "avg_read_wall_ms=%.3f avg_read_cpu_ms=%.3f",
            self.reader_name,
            self.manual_mode,
            elapsed_s,
            self.frame_count,
            frame_rate,
            self.empty_read_count,
            avg_read_wall_ms,
            avg_read_cpu_ms,
        )
        self.reset(now_s)

    def reset(self, started_at_s: float) -> None:
        self.started_at_s = started_at_s
        self.frame_count = 0
        self.empty_read_count = 0
        self.read_wall_time_s = 0.0
        self.read_cpu_time_s = 0.0


@dataclass(kw_only=True)
class ProcessorStats:
    """Interval stats for processor queue, skip, and batch activity."""

    interval_secs: int
    logger: Logger
    started_at_s: float = 0.0
    empty_poll_count: int = 0
    input_frame_count: int = 0
    skipped_frame_count: int = 0
    batch_count: int = 0
    output_frame_count: int = 0
    error_count: int = 0
    batch_wall_time_s: float = 0.0
    batch_cpu_time_s: float = 0.0

    def __post_init__(self) -> None:
        self.reset(time.perf_counter())

    def record_empty_poll(self) -> None:
        self.empty_poll_count += 1
        self.log_if_needed()

    def record_input_frame(self) -> None:
        self.input_frame_count += 1

    def record_skipped_frame(self) -> None:
        self.skipped_frame_count += 1

    def record_error(self) -> None:
        self.error_count += 1
        self.log_if_needed()

    def record_batch(self, batch_size: int, wall_time_s: float, cpu_time_s: float) -> None:
        self.batch_count += 1
        self.output_frame_count += batch_size
        self.batch_wall_time_s += wall_time_s
        self.batch_cpu_time_s += cpu_time_s
        self.log_if_needed()

    def log_if_needed(self) -> None:
        now_s = time.perf_counter()
        elapsed_s = now_s - self.started_at_s
        if elapsed_s < self.interval_secs:
            return

        batch_rate = self.batch_count / elapsed_s if elapsed_s > 0 else 0.0
        output_rate = self.output_frame_count / elapsed_s if elapsed_s > 0 else 0.0
        avg_batch_wall_ms = (self.batch_wall_time_s / self.batch_count) * 1000 if self.batch_count else 0.0
        avg_batch_cpu_ms = (self.batch_cpu_time_s / self.batch_count) * 1000 if self.batch_count else 0.0
        self.logger.info(
            "processor_stats interval_secs=%.1f batches=%d batch_rate=%.1f input_frames=%d skipped_frames=%d "
            "output_frames=%d output_rate=%.1f empty_polls=%d errors=%d avg_batch_wall_ms=%.3f avg_batch_cpu_ms=%.3f",
            elapsed_s,
            self.batch_count,
            batch_rate,
            self.input_frame_count,
            self.skipped_frame_count,
            self.output_frame_count,
            output_rate,
            self.empty_poll_count,
            self.error_count,
            avg_batch_wall_ms,
            avg_batch_cpu_ms,
        )
        self.reset(now_s)

    def reset(self, started_at_s: float) -> None:
        self.started_at_s = started_at_s
        self.empty_poll_count = 0
        self.input_frame_count = 0
        self.skipped_frame_count = 0
        self.batch_count = 0
        self.output_frame_count = 0
        self.error_count = 0
        self.batch_wall_time_s = 0.0
        self.batch_cpu_time_s = 0.0


@dataclass(kw_only=True)
class WebRtcRecvStats:
    """Interval stats for WebRTC track recv calls."""

    interval_secs: int
    logger: Logger
    started_at_s: float = 0.0
    recv_count: int = 0
    new_frame_count: int = 0
    cached_frame_count: int = 0
    fallback_frame_count: int = 0
    error_frame_count: int = 0
    recv_wall_time_s: float = 0.0
    recv_cpu_time_s: float = 0.0

    def __post_init__(self) -> None:
        self.reset(time.perf_counter())

    def record(self, frame_kind: WebRtcFrameKind, wall_time_s: float, cpu_time_s: float) -> None:
        self.recv_count += 1
        self.recv_wall_time_s += wall_time_s
        self.recv_cpu_time_s += cpu_time_s
        match frame_kind:
            case WebRtcFrameKind.NEW:
                self.new_frame_count += 1
            case WebRtcFrameKind.CACHED:
                self.cached_frame_count += 1
            case WebRtcFrameKind.ERROR:
                self.error_frame_count += 1
            case WebRtcFrameKind.FALLBACK:
                self.fallback_frame_count += 1

        now_s = time.perf_counter()
        elapsed_s = now_s - self.started_at_s
        if elapsed_s < self.interval_secs:
            return

        recv_rate = self.recv_count / elapsed_s if elapsed_s > 0 else 0.0
        avg_wall_ms = (self.recv_wall_time_s / self.recv_count) * 1000 if self.recv_count else 0.0
        avg_cpu_ms = (self.recv_cpu_time_s / self.recv_count) * 1000 if self.recv_count else 0.0
        self.logger.info(
            "webrtc_recv_stats interval_secs=%.1f recv_count=%d recv_rate=%.1f new_frames=%d cached_frames=%d "
            "fallback_frames=%d error_frames=%d avg_recv_wall_ms=%.3f avg_recv_cpu_ms=%.3f",
            elapsed_s,
            self.recv_count,
            recv_rate,
            self.new_frame_count,
            self.cached_frame_count,
            self.fallback_frame_count,
            self.error_frame_count,
            avg_wall_ms,
            avg_cpu_ms,
        )
        self.reset(now_s)

    def reset(self, started_at_s: float) -> None:
        self.started_at_s = started_at_s
        self.recv_count = 0
        self.new_frame_count = 0
        self.cached_frame_count = 0
        self.fallback_frame_count = 0
        self.error_frame_count = 0
        self.recv_wall_time_s = 0.0
        self.recv_cpu_time_s = 0.0
