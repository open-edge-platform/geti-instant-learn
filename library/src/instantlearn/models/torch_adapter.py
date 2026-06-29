# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch adapter: the single bridge between backend-neutral data and torch.

This module is the *only* place where ``instantlearn.data.base`` numpy
containers are converted into torch tensors (and back). Keeping the conversion
here (instead of on ``Sample`` / ``Prediction``) preserves dependency
inversion: the backend-neutral abstractions never import torch, and adding a
new backend never forces a change to the core data classes (Open/Closed).

Torch-backed models are the consumers: they call :func:`samples_to_tensors`
to convert inputs and :func:`tensors_to_prediction` (from torch tensors) or
:func:`arrays_to_prediction` (from numpy arrays) to build the numpy
``Prediction`` at the return boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from instantlearn.data.base.batch import Batch
from instantlearn.data.base.prediction import Prediction
from instantlearn.data.base.sample import Sample

if TYPE_CHECKING:
    from collections.abc import Sequence


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


def samples_to_tensors(target: Sample | list[Sample] | Batch, device: str = "cpu") -> list[TensorSample]:
    """Convert ``Sample`` / ``list[Sample]`` / ``Batch`` inputs to ``TensorSample``.

    This is the single numpy->torch entry point for model inputs. Single
    samples, lists, and ``Batch`` objects are handled uniformly.

    Args:
        target: One or more samples, or a ``Batch``.
        device: Target device string, e.g. ``"cpu"`` or ``"cuda"``.

    Returns:
        A list of ``TensorSample`` objects on *device*.
    """
    if isinstance(target, Sample):
        target = [target]
    elif isinstance(target, Batch):
        target = target.samples
    return [sample_to_tensors(s, device) for s in target]


def tensors_to_prediction(
    masks: torch.Tensor,
    scores: torch.Tensor,
    label_ids: torch.Tensor,
    categories: Sequence[str],
    boxes: torch.Tensor | None = None,
    points: torch.Tensor | None = None,
    metadata: dict | None = None,
) -> Prediction:
    """Convert torch model outputs to a numpy ``Prediction``.

    This is the single torch->numpy boundary for all PyTorch-backed models.
    Every tensor is moved to host memory via ``detach().cpu().numpy()``, then
    dtype normalization and ``label_ids -> label_names`` mapping are applied by
    :func:`arrays_to_prediction`. The resulting ``Prediction`` is a pure numpy,
    backend-neutral data container.

    Args:
        masks: Instance masks tensor of shape ``(N, H, W)``.
        scores: Confidence scores tensor of shape ``(N,)``.
        label_ids: Integer category IDs tensor of shape ``(N,)``.
        categories: Sequence mapping a label ID to its category name.
        boxes: Optional bounding boxes tensor of shape ``(N, 4)``.
        points: Optional point predictions tensor of shape ``(N, K, 2)``.
        metadata: Optional free-form per-prediction metadata.

    Returns:
        A numpy ``Prediction`` with contract dtypes enforced.
    """

    def _np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    def _np_opt(t: torch.Tensor | None) -> np.ndarray | None:
        return t.detach().cpu().numpy() if t is not None else None

    return arrays_to_prediction(
        masks=_np(masks),
        scores=_np(scores),
        label_ids=_np(label_ids),
        categories=categories,
        boxes=_np_opt(boxes),
        points=_np_opt(points),
        metadata=metadata,
    )


def arrays_to_prediction(
    masks: np.ndarray,
    scores: np.ndarray,
    label_ids: np.ndarray,
    categories: Sequence[str],
    boxes: np.ndarray | None = None,
    points: np.ndarray | None = None,
    metadata: dict | None = None,
) -> Prediction:
    """Assemble a normalized numpy ``Prediction`` from raw numpy arrays.

    Enforces the contract dtypes:

    - ``masks``: ``bool`` if already boolean, otherwise ``uint8``.
    - ``scores``: ``float32``.
    - ``label_ids``: ``int32``.
    - ``boxes`` / ``points``: ``float32`` when present.

    ``label_names`` is derived by indexing ``categories`` with each entry of
    ``label_ids``; IDs outside the range fall back to ``str(id)``.

    Args:
        masks: Instance masks of shape ``(N, H, W)``.
        scores: Per-instance confidence scores of shape ``(N,)``.
        label_ids: Per-instance integer category IDs of shape ``(N,)``.
        categories: Sequence mapping a label ID to its category name.
        boxes: Optional bounding boxes of shape ``(N, 4)`` in xyxy format.
        points: Optional point predictions of shape ``(N, K, 2)``.
        metadata: Optional free-form per-prediction metadata.

    Returns:
        A ``Prediction`` with all arrays cast to the contract dtypes.
    """
    masks = np.ascontiguousarray(masks)
    if masks.dtype != np.bool_:
        masks = masks.astype(np.uint8, copy=False)

    scores = np.ascontiguousarray(scores, dtype=np.float32)
    label_ids = np.ascontiguousarray(label_ids, dtype=np.int32)

    n_categories = len(categories)
    label_names = np.array(
        [categories[i] if 0 <= i < n_categories else str(i) for i in label_ids.tolist()],
        dtype=object,
    )

    if boxes is not None:
        boxes = np.ascontiguousarray(boxes, dtype=np.float32)
    if points is not None:
        points = np.ascontiguousarray(points, dtype=np.float32)

    return Prediction(
        masks=masks,
        scores=scores,
        label_ids=label_ids,
        label_names=label_names,
        boxes=boxes,
        points=points,
        metadata=metadata if metadata is not None else {},
    )
