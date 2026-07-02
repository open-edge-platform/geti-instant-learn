# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from threading import Event, Lock, Thread

import psutil

logger = logging.getLogger(__name__)

_BYTES_PER_MIB = 1024 * 1024


@dataclass(frozen=True)
class BackendCpuSample:
    pid: int
    process_cpu_percent: float
    system_cpu_percent: float
    rss_mb: float
    thread_count: int


class BackendCpuMonitor:
    """Periodically logs coarse CPU and memory usage for the backend process."""

    def __init__(self, interval_secs: int, process: psutil.Process | None = None) -> None:
        if interval_secs <= 0:
            raise ValueError("interval_secs must be greater than 0")
        self._interval_secs = interval_secs
        self._process = process or psutil.Process()
        self._stop_event = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._latest_sample: BackendCpuSample | None = None

    @property
    def latest_sample(self) -> BackendCpuSample | None:
        with self._lock:
            return self._latest_sample

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.debug("Backend CPU monitoring is already running")
            return

        self._stop_event.clear()
        self._prime_cpu_counters()
        self._thread = Thread(target=self._run, name="backend-cpu-monitor", daemon=True)
        self._thread.start()
        logger.info("Backend CPU monitoring started: interval_secs=%d pid=%d", self._interval_secs, self._process.pid)

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._interval_secs + 1)
            if thread.is_alive():
                logger.warning("Backend CPU monitoring did not stop within timeout")
                return

        self._thread = None
        logger.info("Backend CPU monitoring stopped")

    def sample(self) -> BackendCpuSample:
        memory_info = self._process.memory_info()
        sample = BackendCpuSample(
            pid=self._process.pid,
            process_cpu_percent=self._process.cpu_percent(interval=None),
            system_cpu_percent=psutil.cpu_percent(interval=None),
            rss_mb=memory_info.rss / _BYTES_PER_MIB,
            thread_count=self._process.num_threads(),
        )
        with self._lock:
            self._latest_sample = sample
        return sample

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_secs):
            try:
                sample = self.sample()
            except Exception:
                logger.exception("Failed to collect backend CPU usage sample")
                continue
            self._log_sample(sample)

    def _prime_cpu_counters(self) -> None:
        try:
            self._process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except Exception:
            logger.exception("Failed to initialize backend CPU usage counters")

    @staticmethod
    def _log_sample(sample: BackendCpuSample) -> None:
        logger.info(
            "backend_cpu pid=%d process_cpu_percent=%.1f system_cpu_percent=%.1f rss_mb=%.1f thread_count=%d",
            sample.pid,
            sample.process_cpu_percent,
            sample.system_cpu_percent,
            sample.rss_mb,
            sample.thread_count,
        )
