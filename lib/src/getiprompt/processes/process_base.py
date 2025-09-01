# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for processes."""

from abc import ABC, abstractmethod

from getiprompt.utils.decorators import time_call


class Process(ABC):
    """This is the base class of processes within pipelines.

    Typically classes that inherit from this method implement a __call__ method that
    accepts lists of objects. Each index in the list represents the image that it came from,
    so a List[Features] represents multiple features per image because it was generated using the
    Encoder that used a List[Image] as an input.
    """

    def __init__(self) -> None:
        """This initializes the process."""
        self._last_duration: float = 0.0

    @abstractmethod
    def __call__(self, *args, **kwargs) -> None:
        """This method must be implemented by subclasses of Process."""
        msg = f"The __call__ method must be implemented by subclasses of Process ({self.__class__.__name__})"
        raise NotImplementedError(msg)

    def __init_subclass__(cls, **kwargs) -> None:
        """This method is called when a subclass of Process is defined.

        It decorates the __call__ method of the subclass with the time_call decorator.
        This decorator times the execution of the __call__ method and stores the duration on the instance.
        This is used to print the timing of the pipeline components.
        """
        super().__init_subclass__(**kwargs)
        if callable(cls) and cls.__call__ is not Process.__call__:
            original_call = cls.__call__
            if callable(original_call) and not (
                hasattr(original_call, "__wrapped__")
                and original_call.__wrapped__.__name__ == "wrapper"
                and "time_call" in original_call.__code__.co_filename
            ):
                cls.__call__ = time_call(original_call)
