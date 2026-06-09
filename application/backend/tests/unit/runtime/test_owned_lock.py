#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import threading

import pytest

from runtime.pipeline_manager import _OwnedLock


class TestOwnedLock:
    def test_assert_held_raises_when_not_acquired(self):
        lock = _OwnedLock()
        with pytest.raises(RuntimeError, match="Must be called while the lock is held"):
            lock.assert_held()

    def test_assert_held_passes_inside_with_block(self):
        lock = _OwnedLock()
        with lock:
            lock.assert_held()  # must not raise

    def test_assert_held_raises_after_release(self):
        lock = _OwnedLock()
        with lock:
            pass
        with pytest.raises(RuntimeError, match="Must be called while the lock is held"):
            lock.assert_held()

    def test_assert_held_raises_from_different_thread(self):
        lock = _OwnedLock()
        errors: list[Exception] = []

        def try_assert():
            try:
                lock.assert_held()
            except RuntimeError as exc:
                errors.append(exc)

        with lock:
            t = threading.Thread(target=try_assert)
            t.start()
            t.join()

        assert len(errors) == 1
        assert "Must be called while the lock is held" in str(errors[0])

    def test_context_manager_blocks_second_thread(self):
        lock = _OwnedLock()
        acquired_inside = threading.Event()
        proceed = threading.Event()
        second_acquired = threading.Event()

        def second_thread():
            proceed.wait()
            with lock:
                second_acquired.set()

        t = threading.Thread(target=second_thread)
        t.start()

        with lock:
            acquired_inside.set()
            proceed.set()
            # Give the second thread time to attempt acquisition
            assert not second_acquired.wait(timeout=0.1), "Second thread should not have acquired lock yet"

        t.join(timeout=1)
        assert second_acquired.is_set(), "Second thread should have acquired lock after release"
