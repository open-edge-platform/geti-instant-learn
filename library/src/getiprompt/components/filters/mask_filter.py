# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch import nn
from torchvision.ops import batched_nms, masks_to_boxes

from getiprompt.types import Masks, Points


class ClassOverlapMaskFilter(nn.Module):
    """This filter inspects overlapping areas between different label masks."""

    def forward(
        self,
        all_pred_masks: list[Masks] | None = None,
        all_pred_points: list[Points] | None = None,
        threshold_iou: float = 0.8,
    ) -> tuple[list[Masks], list[Points]]:
        """Inspect overlapping areas between different label masks using NMS.

        Args:
            masks_per_image: Predicted mask for each image and all labels
            used_points_per_image: Used points for each image and all labels
            threshold_iou: Threshold for IOU between the masks

        Returns:
            List of masks per image, list of points per image
        """
        result_masks = []
        result_points = []

        for pred_masks, pred_points in zip(all_pred_masks, all_pred_points, strict=True):
            if not pred_masks.data:
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
                _scores.extend(pred_points.only_foreground().get(label_id, [])[0][:, 2])
                _foreground_points.extend(pred_points.only_foreground().get(label_id, [])[0])
                _labels.extend(torch.full((pred_masks.data[label_id].shape[0],), label_id, device=device))

            if len(_masks) == 0:
                result_masks.append(pred_masks)
                result_points.append(pred_points)
                continue

            _masks = torch.stack(_masks)
            _boxes = masks_to_boxes(_masks)
            _scores = torch.stack(_scores)
            _foreground_points = torch.stack(_foreground_points)
            _labels = torch.stack(_labels)

            keep_indices = batched_nms(_boxes, _labels, _scores, threshold_iou)
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
