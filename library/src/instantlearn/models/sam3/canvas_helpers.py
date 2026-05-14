# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for SAM3 canvas mode.

Pure geometry and image manipulation utilities used by the SAM3 canvas
prediction pipeline. These functions are stateless — they operate on
tensors and arrays without requiring model state.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torchvision.ops import nms as torchvision_nms


def crop_around_bbox(
    image: torch.Tensor,
    bbox: np.ndarray,
    crop_padding: float,
) -> tuple[torch.Tensor, np.ndarray]:
    """Crop image tightly around bbox with padding.

    Args:
        image: (C, H, W) image tensor.
        bbox: [x1, y1, x2, y2] bounding box.
        crop_padding: Padding factor around the bbox. A factor of 2.0 means
            the crop region is 2x the bbox size in each dimension.

    Returns:
        (cropped_image, adjusted_bbox) where adjusted_bbox is in crop coordinates.
    """
    _, h, w = image.shape
    x1, y1, x2, y2 = bbox[:4].astype(float)
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    half_w = bw * crop_padding / 2
    half_h = bh * crop_padding / 2

    crop_x1 = int(max(0, cx - half_w))
    crop_y1 = int(max(0, cy - half_h))
    crop_x2 = int(min(w, cx + half_w))
    crop_y2 = int(min(h, cy + half_h))
    crop_x2 = max(crop_x2, crop_x1 + 1)
    crop_y2 = max(crop_y2, crop_y1 + 1)

    crop = image[:, crop_y1:crop_y2, crop_x1:crop_x2]
    adj_bbox = np.array([
        x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1,
    ], dtype=np.float32)
    return crop, adj_bbox


def build_canvas_vertical(
    ref_image: torch.Tensor,
    tgt_image: torch.Tensor,
    ref_bbox: np.ndarray,
    split_ratio: float,
) -> tuple[torch.Tensor, np.ndarray, tuple[int, int, int, int]]:
    """Build vertical canvas: target on top, reference on bottom.

    Args:
        ref_image: (C, H, W) reference image tensor.
        tgt_image: (C, H, W) target image tensor.
        ref_bbox: [x1, y1, x2, y2] bounding box on the reference image.
        split_ratio: Fraction of canvas height allocated to reference strip.

    Returns:
        (canvas, canvas_bbox, tgt_region) where tgt_region is (x, y, w, h).
    """
    c = ref_image.shape[0]
    ref_h, ref_w = ref_image.shape[1], ref_image.shape[2]
    tgt_w = tgt_image.shape[2]

    canvas_w = max(ref_w, tgt_w)
    canvas_h = max(canvas_w, 2)

    ref_canvas_h = int(canvas_h * split_ratio)
    ref_canvas_h = min(max(ref_canvas_h, 1), canvas_h - 1)
    tgt_canvas_h = canvas_h - ref_canvas_h

    ref_resized = F.interpolate(
        ref_image.unsqueeze(0).float(), size=(ref_canvas_h, canvas_w),
        mode="bilinear", align_corners=False,
    ).squeeze(0)
    tgt_resized = F.interpolate(
        tgt_image.unsqueeze(0).float(), size=(tgt_canvas_h, canvas_w),
        mode="bilinear", align_corners=False,
    ).squeeze(0)

    canvas = torch.zeros(c, canvas_h, canvas_w, dtype=ref_resized.dtype)
    canvas[:, :tgt_canvas_h, :canvas_w] = tgt_resized
    canvas[:, tgt_canvas_h:, :canvas_w] = ref_resized

    sx = canvas_w / ref_w
    sy = ref_canvas_h / ref_h
    x1, y1, x2, y2 = ref_bbox[:4]
    canvas_bbox = np.array([
        x1 * sx, y1 * sy + tgt_canvas_h,
        x2 * sx, y2 * sy + tgt_canvas_h,
    ], dtype=np.float32)

    return canvas, canvas_bbox, (0, 0, canvas_w, tgt_canvas_h)


def build_canvas_multishot(
    ref_images: list[torch.Tensor],
    tgt_image: torch.Tensor,
    ref_bboxes: list[np.ndarray],
    split_ratio: float,
    crop_padding: float,
) -> tuple[torch.Tensor, list[np.ndarray], tuple[int, int, int, int]]:
    """Build multi-shot canvas: multiple cropped references in a strip.

    Args:
        ref_images: List of (C, H, W) reference image tensors.
        tgt_image: (C, H, W) target image tensor.
        ref_bboxes: List of [x1, y1, x2, y2] bounding boxes on references.
        split_ratio: Fraction of canvas height allocated to reference strip.
        crop_padding: Padding factor for cropping around reference bboxes.

    Returns:
        (canvas, canvas_bboxes, tgt_region).

    Raises:
        ValueError: If canvas width is too small for the number of references.
    """
    crops, adj_bboxes = [], []
    for ref_img, ref_bbox in zip(ref_images, ref_bboxes, strict=True):
        crop, adj_bbox = crop_around_bbox(ref_img, ref_bbox, crop_padding)
        crops.append(crop)
        adj_bboxes.append(adj_bbox)

    c = tgt_image.shape[0]
    canvas_w = max(tgt_image.shape[2], *(crop.shape[2] for crop in crops))
    canvas_h = max(canvas_w, 2)

    ref_strip_h = int(canvas_h * split_ratio)
    ref_strip_h = min(max(ref_strip_h, 1), canvas_h - 1)
    tgt_canvas_h = canvas_h - ref_strip_h

    tgt_resized = F.interpolate(
        tgt_image.unsqueeze(0).float(), size=(tgt_canvas_h, canvas_w),
        mode="bilinear", align_corners=False,
    ).squeeze(0)

    n_refs = len(crops)
    if n_refs > canvas_w:
        msg = (
            "Canvas layout requires at least one pixel per reference crop. "
            f"Got canvas width {canvas_w} for {n_refs} reference crops. "
            "Reduce the number of references or increase canvas width."
        )
        raise ValueError(msg)
    crop_w = canvas_w // n_refs
    remainder = canvas_w - crop_w * n_refs

    ref_strip = torch.zeros(c, ref_strip_h, canvas_w, dtype=tgt_resized.dtype)
    canvas_bboxes: list[np.ndarray] = []
    x_offset = 0
    for i, (crop, adj_bbox) in enumerate(zip(crops, adj_bboxes, strict=True)):
        this_w = crop_w + (remainder if i == n_refs - 1 else 0)
        crop_resized = F.interpolate(
            crop.unsqueeze(0).float(), size=(ref_strip_h, this_w),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        ref_strip[:, :, x_offset:x_offset + this_w] = crop_resized

        sx = this_w / crop.shape[2]
        sy = ref_strip_h / crop.shape[1]
        ax1, ay1, ax2, ay2 = adj_bbox[:4]
        canvas_bboxes.append(np.array([
            ax1 * sx + x_offset, ay1 * sy + tgt_canvas_h,
            ax2 * sx + x_offset, ay2 * sy + tgt_canvas_h,
        ], dtype=np.float32))
        x_offset += this_w

    canvas = torch.zeros(c, canvas_h, canvas_w, dtype=tgt_resized.dtype)
    canvas[:, :tgt_canvas_h, :] = tgt_resized
    canvas[:, tgt_canvas_h:, :] = ref_strip

    return canvas, canvas_bboxes, (0, 0, canvas_w, tgt_canvas_h)


def extract_target_predictions(
    pred: dict[str, torch.Tensor],
    tgt_region: tuple[int, int, int, int],
    tgt_h: int,
    tgt_w: int,
) -> dict[str, torch.Tensor]:
    """Extract predictions from the target region and remap to original coords.

    Args:
        pred: Prediction dict with 'pred_boxes' and optionally 'pred_masks'.
        tgt_region: (x, y, w, h) of target region on canvas.
        tgt_h: Original target image height.
        tgt_w: Original target image width.

    Returns:
        Prediction dict with boxes/masks remapped to original target coordinates.
    """
    tx, ty, tw, th = tgt_region
    pred_boxes = pred["pred_boxes"][:, :4].cpu()

    if pred_boxes.shape[0] == 0:
        return {
            "pred_boxes": torch.empty(0, 5),
            "pred_masks": torch.empty(0, tgt_h, tgt_w),
            "pred_labels": torch.empty(0, dtype=torch.int64),
        }

    cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    in_target = (cx >= tx) & (cx < tx + tw) & (cy >= ty) & (cy < ty + th)

    scores = (
        pred["pred_boxes"][:, 4].cpu()
        if pred["pred_boxes"].shape[1] > 4
        else torch.ones(len(pred_boxes))
    )
    target_boxes = pred_boxes[in_target]
    target_scores = scores[in_target]

    result: dict[str, torch.Tensor] = {}

    if target_boxes.shape[0] > 0:
        scale_x = tgt_w / tw
        scale_y = tgt_h / th
        remapped = target_boxes.clone()
        remapped[:, 0] = (target_boxes[:, 0] - tx) * scale_x
        remapped[:, 1] = (target_boxes[:, 1] - ty) * scale_y
        remapped[:, 2] = (target_boxes[:, 2] - tx) * scale_x
        remapped[:, 3] = (target_boxes[:, 3] - ty) * scale_y
        remapped[:, 0].clamp_(min=0)
        remapped[:, 1].clamp_(min=0)
        remapped[:, 2].clamp_(max=tgt_w)
        remapped[:, 3].clamp_(max=tgt_h)
        result["pred_boxes"] = torch.cat([remapped, target_scores.unsqueeze(1)], dim=1)
    else:
        result["pred_boxes"] = torch.empty(0, 5)

    if "pred_masks" in pred and pred["pred_masks"].shape[0] > 0:
        canvas_masks = pred["pred_masks"].cpu()
        target_masks = canvas_masks[in_target]
        if target_masks.shape[0] > 0:
            target_masks = target_masks[:, ty:ty + th, tx:tx + tw]
            target_masks = F.interpolate(
                target_masks.unsqueeze(1).float(),
                size=(tgt_h, tgt_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            result["pred_masks"] = (target_masks > 0.5).to(torch.uint8)
        else:
            result["pred_masks"] = torch.empty(0, tgt_h, tgt_w)
    else:
        result["pred_masks"] = torch.empty(0, tgt_h, tgt_w)

    if "pred_labels" in pred:
        result["pred_labels"] = pred["pred_labels"][in_target].cpu()
    else:
        result["pred_labels"] = torch.zeros(result["pred_boxes"].shape[0], dtype=torch.int64)

    return result


def merge_cross_category(
    boxes_list: list[torch.Tensor],
    masks_list: list[torch.Tensor],
    labels_list: list[torch.Tensor],
    img_size: tuple[int, int],
    iou_threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Merge per-category predictions with cross-category NMS.

    When the same object is detected by multiple category canvases,
    keeps only the highest-confidence prediction.

    Args:
        boxes_list: Per-category box tensors [N_k, 5] (x1,y1,x2,y2,score).
        masks_list: Per-category mask tensors [N_k, H, W].
        labels_list: Per-category label tensors [N_k].
        img_size: (height, width) of the target image.
        iou_threshold: IoU threshold for cross-category NMS.

    Returns:
        Merged prediction dict.
    """
    all_boxes = torch.cat(boxes_list, dim=0)
    all_masks = torch.cat(masks_list, dim=0) if masks_list else torch.empty(0, *img_size)
    all_labels = torch.cat(labels_list, dim=0)

    if all_boxes.shape[0] == 0:
        return {
            "pred_boxes": torch.empty(0, 5),
            "pred_masks": torch.empty(0, *img_size),
            "pred_labels": torch.empty(0, dtype=torch.int64),
        }

    coords = all_boxes[:, :4]
    scores = all_boxes[:, 4]
    keep = torchvision_nms(coords, scores, iou_threshold)

    return {
        "pred_boxes": all_boxes[keep],
        "pred_masks": all_masks[keep] if all_masks.shape[0] > 0 else torch.empty(0, *img_size),
        "pred_labels": all_labels[keep],
    }
