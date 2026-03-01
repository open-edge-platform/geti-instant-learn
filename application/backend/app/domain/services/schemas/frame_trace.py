# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Per-frame tracing context for pipeline latency diagnostics.

A FrameTrace is created at the source, carried through InputData / OutputData,
and logged at the terminal component (Sink or WebRTC).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass(order=True)
class ComponentSpan:
    """Timing span recorded by a single pipeline component.

    Ordered by ``start_ms`` so spans are naturally comparable by
    their chronological position in the pipeline.
    """

    start_ms: float
    component: str = field(compare=False)
    end_ms: float | None = field(default=None, compare=False)

    @property
    def duration_ms(self) -> float | None:
        """Duration in milliseconds, or None if the span is still open."""
        if self.end_ms is None:
            return None
        return self.end_ms - self.start_ms


@dataclass
class FrameTrace:
    """Tracing context attached to every frame flowing through the pipeline.

    Responsibilities:
        - Hold a unique frame identifier.
        - Collect ordered component timing spans.
        - Format a human-readable log line.

    Usage:
        trace = FrameTrace.create()
        trace.record_start("source")
        ...
        trace.record_end("source")
    """

    frame_id: str
    spans: list[ComponentSpan] = field(default_factory=list)
    _open_spans: dict[str, ComponentSpan] = field(default_factory=dict, repr=False)

    @staticmethod
    def create() -> FrameTrace:
        """Factory that generates a new trace with a unique frame ID."""
        return FrameTrace(frame_id=uuid.uuid4().hex)

    @staticmethod
    def _now_ms() -> float:
        """Current monotonic time in milliseconds."""
        return time.monotonic() * 1000.0

    def record_start(self, component: str) -> None:
        """Open a new timing span for *component*."""
        span = ComponentSpan(start_ms=self._now_ms(), component=component)
        self.spans.append(span)
        self._open_spans[component] = span

    def record_end(self, component: str) -> None:
        """Close the open span for *component* in O(1)."""
        span = self._open_spans.pop(component, None)
        if span is not None:
            span.end_ms = self._now_ms()

    def format_log(self) -> str:
        """Format the full trace as a single log line.

        Example output:
            [frame abc123def456] source: 1.23 ms | processor: 45.67 ms | webrtc: 0.89 ms | total: 47.79 ms
        """
        parts: list[str] = [f"[frame {self.frame_id}]"]
        total = 0.0
        for span in self.spans:
            dur = span.duration_ms
            if dur is not None:
                parts.append(f"{span.component}: {dur:.2f} ms")
                total += dur
            else:
                parts.append(f"{span.component}: open")

        if self.spans:
            first_start = self.spans[0].start_ms
            last_span = self.spans[-1]
            if last_span.end_ms is not None:
                wall = last_span.end_ms - first_start
                parts.append(f"wall: {wall:.2f} ms")

        parts.append(f"total: {total:.2f} ms")
        return " | ".join(parts)
