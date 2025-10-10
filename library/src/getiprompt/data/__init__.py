# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Datasets."""

from .base import Dataset
from .dataset_iterator_base import DatasetIter
from .dataset_iterators import (
    BatchedCategoryIter,
    BatchedSingleCategoryIter,
    IndexIter,
)
from .lvis import LVISDataset
from .perseg import PerSegDataset

__all__ = [
    "BatchedCategoryIter",
    "BatchedSingleCategoryIter",
    "Dataset",
    "DatasetIter",
    "IndexIter",
    "LVISDataset",
    "PerSegDataset",
]
