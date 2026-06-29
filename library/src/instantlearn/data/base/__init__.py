# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral data contract.

These classes (``Batch``, ``Category``, ``Collatable``, ``Prediction``,
``Sample``) import zero torch and form the contract every backend shares. The
torch-bound ``Dataset`` and concrete datasets live in
:mod:`instantlearn.data.torch` and must be imported explicitly.
"""

from .batch import Batch, Collatable
from .prediction import Prediction
from .sample import Category, Sample

__all__ = ["Batch", "Category", "Collatable", "Prediction", "Sample"]
