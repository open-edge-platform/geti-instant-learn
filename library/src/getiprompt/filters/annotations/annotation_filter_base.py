# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from getiprompt.filters import Filter
from getiprompt.types import Annotations


class AnnotationFilter(Filter):
    """This is the base class for all annotation filters."""

    def __call__(self, annotations: list[Annotations]) -> list[Annotations]:
        """Filter the annotations."""
