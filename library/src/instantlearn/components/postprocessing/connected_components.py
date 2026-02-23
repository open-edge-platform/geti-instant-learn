# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Connected-component based mask cleaning post-processors.

These processors rely on OpenCV and SciPy, which are NOT ONNX-traceable.
They set ``exportable = False`` and are automatically excluded from
the export graph by :class:`PostProcessorPipeline.exportable_subset`.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from scipy import ndimage

from instantlearn.components.postprocessing.base import PostProcessor


class ConnectedComponentFilter(PostProcessor):
    """Remove small connected components from each mask.

    For each mask independently, finds connected components and keeps
    only those with area >= ``min_component_area``. This precisely
    removes isolated pixel blobs without affecting the main mask shape.

    **Not ONNX-exportable** — uses ``cv2.connectedComponentsWithStats``.

    Args:
        min_component_area: Minimum area (pixels) for a connected
            component to survive. Default: ``100``.
        connectivity: Pixel connectivity, ``4`` or ``8``. Default: ``8``.
    """

    def __init__(self, min_component_area: int = 100, connectivity: int = 8) -> None:
        """Initialize with minimum component area and connectivity."""
        super().__init__()
        self.min_component_area = min_component_area
        self.connectivity = connectivity

    @property
    def exportable(self) -> bool:
        """Not ONNX-exportable (uses OpenCV)."""
        return False

    def forward(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter small connected components from each mask.

        Args:
            masks: Binary masks ``[N, H, W]``.
            scores: Confidence scores ``[N]``.
            labels: Category labels ``[N]``.

        Returns:
            Cleaned (masks, scores, labels). Mask count is unchanged
            but masks may have fewer foreground pixels.
        """
        if masks.size(0) == 0:
            return masks, scores, labels

        device = masks.device
        dtype = masks.dtype
        cleaned_masks = []

        for i in range(masks.size(0)):
            mask_np = masks[i].cpu().numpy().astype(np.uint8)
            num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
                mask_np,
                connectivity=self.connectivity,
            )

            # stats[:, cv2.CC_STAT_AREA] contains areas; label 0 is background
            clean = np.zeros_like(mask_np)
            for lbl in range(1, num_labels):
                if stats[lbl, cv2.CC_STAT_AREA] >= self.min_component_area:
                    clean[label_map == lbl] = 1

            cleaned_masks.append(torch.from_numpy(clean))

        result = torch.stack(cleaned_masks).to(device=device, dtype=dtype)
        return result, scores, labels


class HoleFilling(PostProcessor):
    """Fill enclosed holes inside each mask.

    For each mask, fills background regions that are completely
    enclosed by foreground (holes). Uses ``scipy.ndimage.binary_fill_holes``.

    **Not ONNX-exportable** — uses SciPy.
    """

    @property
    def exportable(self) -> bool:
        """Not ONNX-exportable (uses SciPy)."""
        return False

    def forward(  # noqa: PLR6301
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fill holes in each mask.

        Args:
            masks: Binary masks ``[N, H, W]``.
            scores: Confidence scores ``[N]``.
            labels: Category labels ``[N]``.

        Returns:
            Hole-filled (masks, scores, labels). Mask count is unchanged.
        """
        if masks.size(0) == 0:
            return masks, scores, labels

        device = masks.device
        dtype = masks.dtype
        filled_masks = []

        for i in range(masks.size(0)):
            mask_np = masks[i].cpu().numpy().astype(bool)
            filled = ndimage.binary_fill_holes(mask_np).astype(np.uint8)
            filled_masks.append(torch.from_numpy(filled))

        result = torch.stack(filled_masks).to(device=device, dtype=dtype)
        return result, scores, labels
