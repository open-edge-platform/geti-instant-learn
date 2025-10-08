# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipelines."""

from .grounded_sam_pipeline import GroundedSAM
from .matcher_pipeline import Matcher
from .perdino_pipeline import PerDino
from .persam_pipeline import PerSam
from .pipeline_base import Pipeline
from .pipeline_factory import load_pipeline
from .softmatcher_pipeline import SoftMatcher

__all__ = [
    "Matcher",
    "PerDino",
    "PerSam",
    "Pipeline",
    "SoftMatcher",
    "GroundedSAM",
    "load_pipeline",
]
