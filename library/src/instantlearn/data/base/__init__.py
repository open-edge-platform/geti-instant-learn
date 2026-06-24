# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base classes for datasets.

The backend-neutral contract (``Batch``, ``Category``, ``Collatable``, ``Prediction``, ``Sample``)
is imported eagerly — these modules don't import torch.
``Dataset`` subclasses ``torch.utils.data.Dataset`` and yields ``torch.Tensor`` images, so it is loaded lazily
via :pep:`562`; importing the contract therefore never requires torch.
"""

import importlib
from typing import TYPE_CHECKING

from .batch import Batch, Collatable
from .prediction import Prediction
from .sample import Category, Sample

if TYPE_CHECKING:
    from .base import Dataset

# Maps each torch-bound public name to the submodule that defines it. Kept
# out of the eager imports above so the contract stays torch-free.
_LAZY_MEMBERS = {"Dataset": ".base"}

__all__ = ["Batch", "Category", "Collatable", "Dataset", "Prediction", "Sample"]


def __getattr__(name: str) -> object:
    """Lazily import torch-bound members on first access.

    Raises:
        AttributeError: If *name* is not a public member of this package.
    """
    submodule = _LAZY_MEMBERS.get(name)
    if submodule is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module = importlib.import_module(submodule, __name__)
    return getattr(module, name)
