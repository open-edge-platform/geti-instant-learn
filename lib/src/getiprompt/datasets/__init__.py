# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Datasets."""

from .dataset_base import Dataset
from .dataset_iterator_base import DatasetIter
from .dataset_iterators import (
    BatchedCategoryIter,
    BatchedSingleCategoryIter,
    CategoryIter,
    IndexIter,
)
from .lvis import LVISDataset
from .perseg import PerSegDataset

__all__ = [
    "Dataset",
    "IndexIter",
    "CategoryIter",
    "BatchedSingleCategoryIter",
    "BatchedCategoryIter",
    "DatasetIter",
    "LVISDataset",
    "PerSegDataset",
]
