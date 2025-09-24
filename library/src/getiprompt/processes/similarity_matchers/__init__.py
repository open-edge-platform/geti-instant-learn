# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Similarity matchers."""

from .cosine_similarity import CosineSimilarity
from .similarity_matcher_base import SimilarityMatcher

__all__ = ["CosineSimilarity", "SimilarityMatcher"]
