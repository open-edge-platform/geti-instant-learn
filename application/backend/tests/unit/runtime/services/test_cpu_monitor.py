# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from types import SimpleNamespace

import pytest

from runtime.services import cpu_monitor as cpu_monitor_module
from runtime.services.cpu_monitor import BackendCpuMonitor, BackendCpuSample


class FakeProcess:
    pid = 12345

    def __init__(self) -> None:
        self.cpu_percent_calls = 0

    def cpu_percent(self, interval=None):  # noqa: ANN001, ANN201
        assert interval is None
        self.cpu_percent_calls += 1
        return 12.5

    @staticmethod
    def memory_info():  # noqa: ANN205
        return SimpleNamespace(rss=128 * 1024 * 1024)

    @staticmethod
    def num_threads():  # noqa: ANN205
        return 7


def test_monitor_rejects_invalid_interval():
    with pytest.raises(ValueError, match="interval_secs must be greater than 0"):
        BackendCpuMonitor(interval_secs=0, process=FakeProcess())


def test_sample_collects_backend_process_metrics(mocker):
    process = FakeProcess()
    mocker.patch.object(cpu_monitor_module.psutil, "cpu_percent", return_value=34.5)

    monitor = BackendCpuMonitor(interval_secs=5, process=process)

    sample = monitor.sample()

    assert sample == BackendCpuSample(
        pid=12345,
        process_cpu_percent=12.5,
        system_cpu_percent=34.5,
        rss_mb=128.0,
        thread_count=7,
    )
    assert monitor.latest_sample == sample
    assert process.cpu_percent_calls == 1


def test_log_sample_emits_stable_fields(caplog):
    sample = BackendCpuSample(
        pid=12345,
        process_cpu_percent=12.5,
        system_cpu_percent=34.5,
        rss_mb=128.0,
        thread_count=7,
    )

    with caplog.at_level(logging.INFO, logger=cpu_monitor_module.logger.name):
        BackendCpuMonitor._log_sample(sample)

    assert "backend_cpu" in caplog.text
    assert "pid=12345" in caplog.text
    assert "process_cpu_percent=12.5" in caplog.text
    assert "system_cpu_percent=34.5" in caplog.text
    assert "rss_mb=128.0" in caplog.text
    assert "thread_count=7" in caplog.text


def test_start_and_stop_prime_counters_without_waiting_for_interval(mocker, caplog):
    process = FakeProcess()
    system_cpu_percent = mocker.patch.object(cpu_monitor_module.psutil, "cpu_percent", return_value=34.5)
    monitor = BackendCpuMonitor(interval_secs=30, process=process)

    with caplog.at_level(logging.INFO, logger=cpu_monitor_module.logger.name):
        monitor.start()
        monitor.stop()

    assert process.cpu_percent_calls == 1
    system_cpu_percent.assert_called_once_with(interval=None)
    assert "Backend CPU monitoring started" in caplog.text
    assert "Backend CPU monitoring stopped" in caplog.text


def test_stop_keeps_thread_reference_when_join_times_out(mocker, caplog):
    monitor = BackendCpuMonitor(interval_secs=5, process=FakeProcess())
    thread = mocker.Mock()
    thread.is_alive.return_value = True
    monitor._thread = thread

    with caplog.at_level(logging.WARNING, logger=cpu_monitor_module.logger.name):
        monitor.stop()

    thread.join.assert_called_once_with(timeout=6)
    assert monitor._thread is thread
    assert "Backend CPU monitoring did not stop within timeout" in caplog.text
    assert "Backend CPU monitoring stopped" not in caplog.text
