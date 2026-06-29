# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Prediction dataclass returned by all instantlearn models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class Prediction:
    """Numpy-based prediction container returned by Model.predict().

    PyTorch-backed models convert their tensor outputs via
    torch_adapter.tensors_to_prediction() before returning; OpenVINO-backed
    models construct Prediction directly from their numpy outputs. Both paths
    produce identically shaped, dtype-normalized instances.
    Not frozen — a post-processor pipeline may mutate masks and scores
    in-place to avoid extra memory allocations during processing.

    Attributes:
        masks: Instance masks of shape (N, H, W) bool or uint8.
        scores: Confidence scores of shape (N,) float32.
        label_ids: Integer category IDs of shape (N,) int32.
        label_names: Category names of shape (N,) as a string/object array.
        boxes: Optional bounding boxes of shape (N, 4) float32 in xyxy
            format.
        points: Optional point predictions of shape (N, K, 2) float32.
        metadata: Free-form dict for any additional per-prediction data.
    """

    masks: np.ndarray
    scores: np.ndarray
    label_ids: np.ndarray
    label_names: np.ndarray
    boxes: np.ndarray | None = None
    points: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
