#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from multiprocessing import Event
from types import TracebackType
from typing import Any, TypeVar

import torch
from getiprompt.data.base.batch import Batch

from domain.services.schemas.processor import InputData
from domain.services.schemas.reader import FrameListResponse
from runtime.core.components.errors import UnsupportedOperationError

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
    """An abstract interface for reading frames from various sources.

    All StreamReader implementations MUST conform to the following data contract:

    Frame Format Contract:
    - Color Format: RGB (Red, Green, Blue order)
    - Array Format: HWC (Height, Width, Channels)
    - Data Type: numpy.uint8
    - Value Range: 0-255
    - Shape: (H, W, 3) for color images

    Example:
        frame.shape = (H, W, 3)  # Height=H, Width=W, Channels=3 (RGB)
        frame.dtype = numpy.uint8
        frame[0, 0, 0] = R value (0-255)
        frame[0, 0, 1] = G value (0-255)
        frame[0, 0, 2] = B value (0-255)

    """

    def connect(self) -> None:
        pass

    @abstractmethod
    def read(self) -> InputData | None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "StreamReader":
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None, /
    ) -> None:
        self.close()

    def seek(self, index: int) -> None:  # noqa: ARG002
        """
        Jump to a specific frame index.
        """
        raise UnsupportedOperationError

    def index(self) -> int:
        """
        Get the current frame position.
        """
        raise UnsupportedOperationError

    def list_frames(self, offset: int = 0, limit: int = 30) -> FrameListResponse:  # noqa: ARG002
        """
        Get a paginated list of all available frames.
        """
        raise UnsupportedOperationError


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


class ModelHandler(ABC):
    @abstractmethod
    def initialise(self) -> None:
        pass

    @abstractmethod
    def infer(self, batch: Batch) -> list[dict[str, torch.Tensor]]:
        pass
