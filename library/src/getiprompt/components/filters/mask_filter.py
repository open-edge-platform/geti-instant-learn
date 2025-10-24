# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch import nn
from torchvision.ops import batched_nms, masks_to_boxes

from getiprompt.types import Masks, Points


class ClassOverlapMaskFilter(nn.Module):
    """This filter inspects overlapping areas between different label masks."""

    def __init__(self, threshold_iou: float = 0.8) -> None:
        """Initialize the filter with a threshold for IoU.

        Args:
            threshold_iou: Threshold for IoU between masks for NMS filtering.
        """
        super().__init__()
        self.threshold_iou = threshold_iou

    def forward(
        self,
        all_pred_masks: list[Masks] | None = None,
        all_pred_points: list[Points] | None = None,
        threshold_iou: float | None = None,
    ) -> tuple[list[Masks], list[Points]]:
        """Inspect overlapping areas between different label masks using NMS.

        Args:
            masks_per_image: Predicted mask for each image and all labels
            used_points_per_image: Used points for each image and all labels
            threshold_iou: Threshold for IOU between the masks. If None, uses the instance threshold.

        Returns:
            List of masks per image, list of points per image
        """
        # Use provided threshold or fall back to instance threshold
        if threshold_iou is None:
            threshold_iou = self.threshold_iou

        # Handle None inputs
        if all_pred_masks is None or all_pred_points is None:
            return [], []

        result_masks = []
        result_points = []

        for pred_masks, pred_points in zip(all_pred_masks, all_pred_points, strict=True):
            if not pred_masks.data:
                # If no masks, return empty results for this image
                result_masks.append(Masks())
                result_points.append(Points())
                continue

            # Collect all masks, scores, and track their class membership
            _masks = []
            _scores = []
            _foreground_points = []
            _labels = []

            label_ids = pred_masks.data.keys()
            for label_id in label_ids:
                if len(pred_masks.data[label_id]) == 0:
                    continue
                device = pred_masks.data[label_id].device
                _masks.extend(pred_masks.data[label_id])

                # Get foreground points for this class
                foreground_points = pred_points.only_foreground().get(label_id, [])
                if len(foreground_points) > 0 and len(foreground_points[0]) > 0:
                    _scores.extend(foreground_points[0][:, 2])
                    _foreground_points.extend(foreground_points[0])
                else:
                    # If no foreground points, use dummy scores
                    num_masks = len(pred_masks.data[label_id])
                    _scores.extend([0.5] * num_masks)  # Default score
                    # Create dummy foreground points
                    dummy_points = torch.zeros((num_masks, 4), device=device)
                    dummy_points[:, 2] = 0.5  # Set score
                    dummy_points[:, 3] = 1  # Set label to foreground
                    _foreground_points.extend(dummy_points)

                _labels.extend(torch.full((pred_masks.data[label_id].shape[0],), label_id, device=device))

            if len(_masks) == 0:
                # If no masks, return empty results
                result_masks.append(Masks())
                result_points.append(Points())
                continue

            _masks = torch.stack(_masks)
            _boxes = masks_to_boxes(_masks).to(torch.float32)
            _scores = torch.tensor(_scores, device=_masks.device, dtype=torch.float32)
            _foreground_points = torch.cat(_foreground_points, dim=0)
            _labels = torch.stack(_labels).to(torch.long)

            keep_indices = batched_nms(_boxes, _scores, _labels, threshold_iou)
            _masks = _masks[keep_indices]
            _foreground_points = _foreground_points[keep_indices]
            _labels = _labels[keep_indices]

            image_masks = Masks()
            image_points = Points()
            for mask, foreground_point, label in zip(_masks, _foreground_points, _labels, strict=True):
                image_masks.add(mask, label.item())
                image_points.add(foreground_point, label.item())

            result_masks.append(image_masks)
            result_points.append(image_points)

        return result_masks, result_points
