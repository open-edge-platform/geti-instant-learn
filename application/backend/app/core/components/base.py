#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from multiprocessing import Event
from types import TracebackType
from typing import Any, TypeVar

IN = TypeVar("IN")
OUT = TypeVar("OUT")


class PipelineComponent(ABC):
    """
    An abstract base class for a runnable pipeline component that can be executed in a thread or process. Its lifecycle
    is managed by a stop_event. Subclasses should monitor this event and gracefully terminate their main loop when the
    event is set.
    """

    def __init__(self):
        self._stop_event = Event()

    def __call__(self, *args, **kwargs):  # noqa: ARG002
        # signature must match interface
        self.run()

    @abstractmethod
    def run(self) -> None:
        """The core logic of the component."""

    def _stop(self) -> None:
        pass

    def stop(self) -> None:
        self._stop_event.set()
        self._stop()


class StreamReader(AbstractContextManager, ABC):
    """An abstract interface for reading frames from various sources."""

    @abstractmethod
    def read(self) -> Any | None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "StreamReader":
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None, /
    ) -> None:
        self.close()


class StreamWriter(AbstractContextManager, ABC):
    """An abstract interface for writing processed frames to various sinks."""

    @abstractmethod
    def write(self, data: Any) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "StreamWriter":
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None, /
    ) -> None:
        self.close()
