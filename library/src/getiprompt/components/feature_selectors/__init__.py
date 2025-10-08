# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Feature selectors."""

from getiprompt.components.feature_selectors.all_features import AllFeaturesSelector
from getiprompt.components.feature_selectors.average_features import AverageFeatures
from getiprompt.components.feature_selectors.cluster_features import ClusterFeatures
from getiprompt.components.feature_selectors.base import (
    FeatureSelector,
)

__all__ = [
    "AverageFeatures",
    "ClusterFeatures",
    "FeatureSelector",
    "AllFeaturesSelector",
]
