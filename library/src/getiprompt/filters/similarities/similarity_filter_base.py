# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from getiprompt.filters import Filter
from getiprompt.types import Similarities


class SimilarityFilter(Filter):
    """This is the base class for all similarity filters."""

    def __call__(self, priors: list[Similarities]) -> list[Similarities]:
        """Filter the similarities."""
        return priors
