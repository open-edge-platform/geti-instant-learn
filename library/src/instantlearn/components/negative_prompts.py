# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utilities for converting negative masks into negative point prompts."""

import torch
from torch import nn


class NegativeMaskToPoints(nn.Module):
    """Convert negative (background) masks to point prompts for SAM.

    Samples representative points from negative masks using a centroid +
    sparse-grid strategy.  All operations are pure-tensor so the module is
    ONNX-exportable.

    Args:
        num_points_per_mask: Maximum points to sample from each mask. Default: 5.
    """

    def __init__(self, num_points_per_mask: int = 5) -> None:
        super().__init__()
        self.num_points_per_mask = num_points_per_mask

    @staticmethod
    def _sample_points_from_mask(mask: torch.Tensor, num_points: int) -> torch.Tensor:
        """Sample representative points from a single binary mask.

        Uses centroid as first point, then furthest-point sampling for the rest.

        Args:
            mask: Binary mask (H, W).
            num_points: Maximum number of points to sample.

        Returns:
            Points (M, 2) in (x, y) image coordinates, where M <= num_points.
        """
        coords = mask.nonzero(as_tuple=False)  # (N, 2) with (row, col)
        if coords.shape[0] == 0:
            return torch.empty(0, 2, device=mask.device, dtype=torch.float32)

        coords_float = coords.float()  # (N, 2) row=y, col=x

        # Start with centroid
        centroid = coords_float.mean(dim=0, keepdim=True)  # (1, 2)
        selected = [centroid]

        remaining = num_points - 1
        if remaining > 0 and coords_float.shape[0] > 1:
            # Furthest-point sampling
            dists = torch.cdist(coords_float, centroid).squeeze(-1)  # (N,)
            for _ in range(min(remaining, coords_float.shape[0] - 1)):
                idx = dists.argmax()
                pt = coords_float[idx : idx + 1]  # (1, 2)
                selected.append(pt)
                new_dists = torch.cdist(coords_float, pt).squeeze(-1)
                dists = torch.minimum(dists, new_dists)

        points_yx = torch.cat(selected, dim=0)  # (M, 2) in (y, x)
        # Convert to (x, y) convention
        points_xy = points_yx.flip(dims=[-1])
        return points_xy

    def forward(self, negative_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert negative masks to negative point prompts.

        Args:
            negative_masks: (N, H, W) binary masks of regions to exclude.

        Returns:
            Tuple of:
                points: (M, 2) point coordinates in (x, y) image space.
                labels: (M,) all zeros (background label for SAM).
        """
        all_points: list[torch.Tensor] = []
        for mask in negative_masks:
            pts = self._sample_points_from_mask(mask, self.num_points_per_mask)
            all_points.append(pts)

        if all_points:
            points = torch.cat(all_points, dim=0)
        else:
            points = torch.empty(0, 2, device=negative_masks.device, dtype=torch.float32)

        labels = torch.zeros(points.shape[0], device=points.device, dtype=torch.long)
        return points, labels
