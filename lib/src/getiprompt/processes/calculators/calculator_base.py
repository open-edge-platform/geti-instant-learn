# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for calculators."""

from abc import abstractmethod

from getiprompt.processes import Process


class Calculator(Process):
    """This is the base class for calculators.

    Examples:
        >>> from getiprompt.processes.calculators import Calculator
        >>>
        >>> # As Calculator is an abstract class, you must subclass it.
        >>> class MyCalculator(Calculator):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.value = 0
        ...     def __call__(self, value_to_add: int):
        ...         self.value += value_to_add
        ...
        >>> my_calculator = MyCalculator()
        >>> my_calculator(5)
        >>> my_calculator.value
        5
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, *args, **kwargs) -> None:
        """Calculate the metrics."""
