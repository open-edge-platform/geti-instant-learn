# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""GetiPrompt datasets.

This module provides dataset classes for GetiPrompt tasks.
"""

from .base import GetiPromptDataset
from .lvis import LVISDataset
from .per_seg import PerSegDataset

__all__ = [
    "GetiPromptDataset",
    "LVISDataset",
    "PerSegDataset",
]
