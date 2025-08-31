#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from multiprocessing import Event
from typing import Any, Generic

from .types import IN, OUT, ConfigDict


class PipelineComponent(ABC):

    def __init__(self):
        self._stop_event = Event()

    @abstractmethod
    def run(self) -> None:
        """The core logic of the component, to be implemented by subclasses."""

    def stop(self) -> None:
        """Signals the component's main loop to terminate."""
        self._stop_event.set()


class Processor(Generic[IN, OUT], ABC):

    @abstractmethod
    def get_config(self) -> ConfigDict: pass

    @abstractmethod
    def process(self, input_data: IN) -> OUT: pass


class StreamReader(AbstractContextManager, ABC):

    @abstractmethod
    def get_config(self) -> ConfigDict: pass

    @abstractmethod
    def read(self) -> Any | None: pass

    def close(self) -> None: pass

    def __enter__(self) -> "StreamReader": return self

    def __exit__(self, exc_type, exc_val, exc_tb): self.close()


class StreamWriter(AbstractContextManager, ABC):

    @abstractmethod
    def get_config(self) -> ConfigDict: pass

    @abstractmethod
    def write(self, data: Any) -> None: pass

    def close(self) -> None: pass

    def __enter__(self) -> "StreamWriter": return self

    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
