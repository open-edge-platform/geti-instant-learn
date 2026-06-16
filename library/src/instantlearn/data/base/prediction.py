# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Prediction dataclass returned by all instantlearn models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


@dataclass
class Prediction:
    """Numpy-based prediction container returned by :meth:`Model.predict`.

    Not frozen — a post-processor pipeline may mutate arrays in-place during processing
    to avoid extra memory allocations (important for real-time video with large masks).

    Attributes:
        masks: Binary or soft masks of shape ``(N, H, W)`` bool/uint8.
        scores: Confidence scores of shape ``(N,)`` float32.
        label_ids: Integer category IDs of shape ``(N,)`` int32.
        label_names: String category names of shape ``(N,)`` dtype object.
        boxes: Optional bounding boxes ``(N, 4)`` xyxy float32.
        points: Optional point predictions ``(N, K, 2)`` float32.
        metadata: Free-form dict for any additional per-prediction data.
    """

    masks: np.ndarray
    scores: np.ndarray
    label_ids: np.ndarray
    label_names: np.ndarray
    boxes: np.ndarray | None = None
    points: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def to_tensors(self, device: str = "cpu") -> dict[str, torch.Tensor]:
        """Convert numpy arrays to torch tensors.

        Args:
            device: Target device string, e.g. ``"cpu"`` or ``"cuda"``.

        Returns:
            Dict with keys ``"masks"``, ``"scores"``, ``"label_ids"``, and optionally ``"boxes"`` and ``"points"``.
        """
        import torch  # noqa: PLC0415

        out: dict[str, torch.Tensor] = {
            "masks": torch.from_numpy(self.masks).to(device),
            "scores": torch.from_numpy(self.scores).to(device),
            "label_ids": torch.from_numpy(self.label_ids).to(device),
        }
        if self.boxes is not None:
            out["boxes"] = torch.from_numpy(self.boxes).to(device)
        if self.points is not None:
            out["points"] = torch.from_numpy(self.points).to(device)
        return out
