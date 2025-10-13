# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import torch
from torch import nn

from getiprompt.types import Masks, Points
from getiprompt.utils import calculate_mask_iou


class ClassOverlapMaskFilter(nn.Module):
    """This filter inspects overlapping areas between different label masks."""

    def forward(
        self,
        masks_per_image: list[Masks] | None = None,
        used_points_per_image: list[Points] | None = None,
        threshold_iou: float = 0.8,
    ) -> list[Masks]:
        """Inspect overlapping areas between different label masks.

        Args:
            masks_per_image: Predicted mask for each image and all labels
            used_points_per_image: Used points for each image and all labels
            threshold_iou: Threshold for IOU between the masks

        Returns:
            List of masks per image
        """
        if used_points_per_image is None:
            used_points_per_image = []
        if masks_per_image is None:
            masks_per_image = []
        for image_masks, image_used_points in zip(masks_per_image, used_points_per_image, strict=False):
            for (label, masks), (other_label, other_masks) in product(
                image_masks.data.items(),
                image_masks.data.items(),
            ):
                if other_label <= label:
                    continue
                foreground_point_scores = image_used_points.only_foreground()[label][0][:, 2]
                other_foreground_point_scores = image_used_points.only_foreground()[other_label][0][:, 2]

                overlapped_label = []
                overlapped_other_label = []
                for (im, mask), (jm, other_mask) in product(enumerate(masks), enumerate(other_masks)):
                    mask_iou, intersection = calculate_mask_iou(mask, other_mask)
                    if mask_iou > threshold_iou:
                        if foreground_point_scores[im] > other_foreground_point_scores[jm]:
                            overlapped_other_label.append(jm)
                        else:
                            overlapped_label.append(im)
                    elif mask_iou > 0:
                        # refine the slightly overlapping region
                        overlapped_coords = torch.where(intersection)
                        if foreground_point_scores[im] > other_foreground_point_scores[jm]:
                            other_mask[overlapped_coords] = 0.0
                        else:
                            mask[overlapped_coords] = 0.0

                # Remove masks / points flagged as overlapped in a single operation
                if overlapped_label:
                    keep = torch.ones(masks.size(0), dtype=torch.bool, device=masks.device)
                    keep[list(set(overlapped_label))] = False  # masks to drop
                    image_masks.data[label] = masks[keep]
                    image_used_points.data[label][0] = image_used_points.data[label][0][keep]

                if overlapped_other_label:
                    keep_other = torch.ones(other_masks.size(0), dtype=torch.bool, device=other_masks.device)
                    keep_other[list(set(overlapped_other_label))] = False
                    image_masks.data[other_label] = other_masks[keep_other]
                    image_used_points.data[other_label][0] = image_used_points.data[other_label][0][keep_other]

        return masks_per_image
