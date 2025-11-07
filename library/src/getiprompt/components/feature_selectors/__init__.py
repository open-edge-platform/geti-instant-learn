# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Feature selectors."""

from .average_features import AverageFeatures
from .base import FeatureSelector

__all__ = [
    "AverageFeatures",
    "FeatureSelector",
]
