# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Datasets.

The backend-neutral contract (``Batch``, ``Sample``) is re-exported eagerly
and stays torch-free. The concrete datasets and transforms are torch-bound,
so they are loaded lazily via :pep:`562`; this keeps ``import instantlearn.data``
(and importing the contract submodules) free of torch.
"""

import importlib
from typing import TYPE_CHECKING

from .base import Batch, Sample

if TYPE_CHECKING:
    from .base import Dataset
    from .coco import COCODataset
    from .folder import FolderDataset
    from .lvis import LVISAnnotationMode, LVISDataset
    from .per_seg import PerSegDataset
    from .transforms import ResizeLongestSide, ToTensor

#: Maps each torch-bound public name to the submodule that defines it.
_LAZY_MEMBERS = {
    "Dataset": ".base",
    "COCODataset": ".coco",
    "FolderDataset": ".folder",
    "LVISAnnotationMode": ".lvis",
    "LVISDataset": ".lvis",
    "PerSegDataset": ".per_seg",
    "ResizeLongestSide": ".transforms",
    "ToTensor": ".transforms",
}

__all__ = [
    "Batch",
    "COCODataset",
    "Dataset",
    "FolderDataset",
    "LVISAnnotationMode",
    "LVISDataset",
    "PerSegDataset",
    "ResizeLongestSide",
    "Sample",
    "ToTensor",
]


def __getattr__(name: str) -> object:
    """Lazily import torch-bound members on first access (see :pep:`562`).

    Raises:
        AttributeError: If *name* is not a public member of this package.
    """
    submodule = _LAZY_MEMBERS.get(name)
    if submodule is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module = importlib.import_module(submodule, __name__)
    return getattr(module, name)
