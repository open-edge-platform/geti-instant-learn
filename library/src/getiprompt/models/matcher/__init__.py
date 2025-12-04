# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Matcher model."""

from .inference import InferenceModel as InferenceMatcher
from .matcher import Matcher

__all__ = [
    "InferenceMatcher",
    "Matcher",
]
