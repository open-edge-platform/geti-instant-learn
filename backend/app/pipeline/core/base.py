#  Copyright (C) 2022-2025 Intel Corporation
#  LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic

from .types import IN, OUT


class PipelineComponent(ABC, Callable[[], None]):
    """
    An abstract base class for a callable pipeline component.
    It is a callable object with its own lifecycle, managed externally.
    """

    def __init__(self, name: str):
        self._name = name
        self._stop_event = threading.Event()

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def _main_loop(self) -> None:
        """The core logic of the component, to be implemented by subclasses."""

    def __call__(self) -> None:
        """Makes the component instance callable and resets its state for restarts."""
        self._stop_event.clear()
        print(f"[{self.name}] Starting...")
        self._main_loop()
        print(f"[{self.name}] Stopped.")

    def stop(self) -> None:
        """Signals the component's main loop to terminate."""
        print(f"[{self.name}] Stop signal received.")
        self._stop_event.set()


class Processor(Generic[IN, OUT], ABC):
    @abstractmethod
    def process(self, input_data: IN) -> OUT: pass


class StreamReader(ABC):
    @abstractmethod
    def read(self) -> Any | None: pass

    def close(self) -> None: pass

    def __enter__(self) -> "StreamReader": return self

    def __exit__(self, exc_type, exc_val, exc_tb): self.close()


class StreamWriter(ABC):
    @abstractmethod
    def write(self, data: Any) -> None: pass

    def close(self) -> None: pass
