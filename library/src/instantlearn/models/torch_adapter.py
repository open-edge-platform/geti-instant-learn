# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch adapter: the single bridge between backend-neutral data and torch.

This module is the *only* place where ``instantlearn.data.base`` numpy
containers are converted into torch tensors. Keeping the conversion here
(instead of on ``Sample`` / ``Prediction``) preserves dependency inversion:
the backend-neutral abstractions never import torch, and adding a new backend
never forces a change to the core data classes (Open/Closed).

``TorchModel`` is the only consumer — it calls :func:`sample_to_tensors` in
``_prepare()`` and builds the numpy ``Prediction`` at the return boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from instantlearn.data.base.sample import Sample


@dataclass
class TensorSample:
    """Torch-native counterpart of :class:`~instantlearn.data.base.sample.Sample`.

    Produced by :func:`sample_to_tensors` and consumed internally by
    ``TorchModel`` subclasses. All array fields are tensors; ``category_labels``
    stays as a plain list of strings.

    Attributes:
        image: Image tensor of shape ``(C, H, W)`` float32.
        masks: Instance masks of shape ``(N, H, W)``.
        bboxes: Bounding boxes of shape ``(N, 4)`` float32 in xyxy format.
        points: Prompt points of shape ``(N, K, 2)`` float32.
        scores: Per-instance scores of shape ``(N,)`` float32.
        category_labels: List of category name strings.
        label_ids: Category IDs of shape ``(N,)`` int32.
    """

    image: torch.Tensor | None = None
    masks: torch.Tensor | None = None
    bboxes: torch.Tensor | None = None
    points: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    category_labels: list[str] | None = None
    label_ids: torch.Tensor | None = None


def sample_to_tensors(sample: Sample, device: str = "cpu") -> TensorSample:
    """Convert a numpy :class:`Sample` to a torch :class:`TensorSample`.

    ``image`` is permuted from HWC to CHW and cast to float32. This is the
    torch boundary — ``Sample`` itself never imports torch.

    Args:
        sample: Backend-neutral numpy sample.
        device: Target device string, e.g. ``"cpu"`` or ``"cuda"``.

    Returns:
        A ``TensorSample`` with all non-``None`` fields moved to *device*.
    """
    image_t = None
    if sample.image is not None:
        arr = sample.image
        if arr.ndim == 3:
            arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        image_t = torch.from_numpy(np.ascontiguousarray(arr)).float().to(device)

    label_ids = sample.label_ids
    return TensorSample(
        image=image_t,
        masks=torch.from_numpy(sample.masks).to(device) if sample.masks is not None else None,
        bboxes=torch.from_numpy(sample.bboxes).float().to(device) if sample.bboxes is not None else None,
        points=torch.from_numpy(sample.points).float().to(device) if sample.points is not None else None,
        scores=torch.from_numpy(sample.scores).float().to(device) if sample.scores is not None else None,
        category_labels=sample.category_labels,
        label_ids=torch.tensor(label_ids, dtype=torch.int32, device=device) if label_ids else None,
    )
