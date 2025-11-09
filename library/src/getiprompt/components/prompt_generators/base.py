# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base classes for prompt generators."""

from torch import nn


class PromptGenerator(nn.Module):
    """This class generates priors.

    Examples:
        >>> from getiprompt.processes.prompt_generators import PromptGenerator
        >>> from getiprompt.types import Priors
        >>>
        >>> class MyPromptGenerator(PromptGenerator):
       ...     def forward(self) -> list[Priors]:
        ...         return [Priors()]
        ...
        >>> my_prompt_generator = MyPromptGenerator()
        >>> priors = my_prompt_generator()
        >>> isinstance(priors[0], Priors)
        True
    """
