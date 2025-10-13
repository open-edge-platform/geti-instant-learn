# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test fixtures for feature selectors."""

import torch

from getiprompt.types import Features


def sample_features() -> Features:
    """Create sample features for testing.

    Returns:
        Features: A Features object with all zero values.
    """
    features = Features()
    features.global_features = torch.randn(5, 4)
    features.local_features = {
        1: [torch.randn(2, 4), torch.randn(1, 4)],
        2: [torch.randn(3, 4)],
    }
    return features


def sample_features_list() -> list[Features]:
    """Create a list of sample features for testing.

    Returns:
        list[Features]: A list of Features objects with all zero values.
    """
    features1 = Features()
    features1.global_features = torch.randn(5, 4)
    features1.local_features = {
        1: [torch.randn(2, 4), torch.randn(1, 4)],
        2: [torch.randn(3, 4)],
    }

    features2 = Features()
    features2.global_features = torch.randn(3, 4)
    features2.local_features = {
        1: [torch.randn(1, 4)],
        3: [torch.randn(2, 4), torch.randn(1, 4)],
    }

    return [features1, features2]


def empty_features() -> Features:
    """Create empty features for testing.

    Returns:
        Features: A Features object with all zero values.
    """
    features = Features()
    features.global_features = torch.randn(0, 4)
    features.local_features = {}
    return features


def single_class_features() -> Features:
    """Create features with single class for testing.

    Returns:
        Features: A Features object with all zero values.
    """
    features = Features()
    features.global_features = torch.randn(3, 4)
    features.local_features = {1: [torch.randn(2, 4), torch.randn(1, 4)]}
    return features


def multi_class_features() -> Features:
    """Create features with multiple classes for testing.

    Returns:
        Features: A Features object with all zero values.
    """
    features = Features()
    features.global_features = torch.randn(5, 4)
    features.local_features = {
        1: [torch.randn(2, 4), torch.randn(1, 4)],
        2: [torch.randn(3, 4)],
        3: [torch.randn(1, 4)],
    }
    return features


def cuda_features() -> Features:
    """Create features on CUDA device for testing.

    Returns:
        Features: A Features object with all zero values.
    """
    if not torch.cuda.is_available():
        return sample_features()

    features = Features()
    features.global_features = torch.randn(5, 4).cuda()
    features.local_features = {
        1: [torch.randn(2, 4).cuda(), torch.randn(1, 4).cuda()],
        2: [torch.randn(3, 4).cuda()],
    }
    return features


def large_features() -> Features:
    """Create large features for performance testing.

    Returns:
        Features: A Features object with all zero values.
    """
    features = Features()
    features.global_features = torch.randn(100, 512)
    features.local_features = {
        1: [torch.randn(50, 512) for _ in range(10)],
        2: [torch.randn(30, 512) for _ in range(5)],
    }
    return features


def zero_features() -> Features:
    """Create features with zero values for testing.

    Returns:
        Features: A Features object with all zero values.
    """
    features = Features()
    features.global_features = torch.zeros(3, 4)
    features.local_features = {
        1: [torch.zeros(2, 4), torch.zeros(1, 4)],
    }
    return features
