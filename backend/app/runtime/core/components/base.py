#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from multiprocessing import Event
from typing import Any, TypeVar

IN = TypeVar("IN")
OUT = TypeVar("OUT")


class JobComponent(ABC):

    def __init__(self):
        self._stop_event = Event()

    def __call__(self, *args, **kwargs):
        self.run()

    @abstractmethod
    def run(self) -> None:
        """The core logic of the component."""

    def _stop(self) -> None:
        pass

    def stop(self) -> None:
        self._stop_event.set()
        self._stop()


class Processor[IN, OUT](ABC):

    @abstractmethod
    def process(self, input_data: IN) -> OUT: pass


class StreamReader(AbstractContextManager, ABC):

    @abstractmethod
    def read(self) -> Any | None: pass

    def close(self) -> None: pass

    def __enter__(self) -> "StreamReader": return self

    def __exit__(self, exc_type, exc_val, exc_tb): self.close()


class StreamWriter(AbstractContextManager, ABC):

    @abstractmethod
    def write(self, data: Any) -> None: pass

    def close(self) -> None: pass

    def __enter__(self) -> "StreamWriter": return self

    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
