# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Traceable SAM decoder for ONNX/TorchScript export."""

import torch
from torch import nn
from torch.nn import functional
from torchvision.ops import nms

from instantlearn.components.sam.predictor import SAMPredictor


def masks_to_boxes_traceable(masks: torch.Tensor) -> torch.Tensor:
    """Traceable version of masks_to_boxes that works with ONNX export.

    Args:
        masks: Tensor[N, H, W] - binary masks

    Returns:
        Tensor[N, 4] - bounding boxes in (x1, y1, x2, y2) format
    """
    # Handle empty masks by clamping to at least 1 element to avoid conditionals
    # The caller should check for empty masks separately
    n = masks.size(0)
    h = masks.size(1)
    w = masks.size(2)

    # Create coordinate grids [N, H, W]
    y_coords = torch.arange(h, device=masks.device, dtype=torch.float).view(1, h, 1).expand(n, h, w)
    x_coords = torch.arange(w, device=masks.device, dtype=torch.float).view(1, 1, w).expand(n, h, w)

    # Boolean mask
    mask_bool = masks.bool()

    # Use large/small sentinel values for min/max reduction
    large_val = torch.tensor(max(h, w) + 1, dtype=torch.float, device=masks.device)

    # For min: non-mask pixels get large value, for max: get -1
    x_for_min = torch.where(mask_bool, x_coords, large_val)
    x_for_max = torch.where(mask_bool, x_coords, -1.0)
    y_for_min = torch.where(mask_bool, y_coords, large_val)
    y_for_max = torch.where(mask_bool, y_coords, -1.0)

    # Flatten spatial dims and reduce
    x1 = x_for_min.flatten(1).min(dim=1).values
    y1 = y_for_min.flatten(1).min(dim=1).values
    x2 = x_for_max.flatten(1).max(dim=1).values
    y2 = y_for_max.flatten(1).max(dim=1).values

    return torch.stack([x1, y1, x2, y2], dim=1)


class SamDecoder(nn.Module):
    """Traceable SAM decoder that accepts tensor inputs for ONNX/TorchScript export.

    This decoder processes point prompts and similarities as tensors instead of
    dictionaries, enabling full pipeline traceability.

    Args:
        sam_predictor: PyTorch SAM predictor instance.
        target_length: Target length for image preprocessing. Default: 1024.
        confidence_threshold: Minimum confidence score for keeping predicted masks
        in the final output. Higher values = stricter filtering. Default: 0.38.
        nms_iou_threshold: IoU threshold for NMS. Default: 0.1.
        max_masks_per_category: Maximum masks to return per category (for padding). Default: 10.
        use_mask_refinement: Whether to use 2-stage mask refinement with box prompts. Default: False.
        merge_masks_per_class: Whether to merge all masks into single mask per class. Default: True.
        use_nms: Whether to use NMS when predicting masks. Default: True.
    """

    def __init__(
        self,
        sam_predictor: SAMPredictor,
        confidence_threshold: float = 0.38,
        nms_iou_threshold: float = 0.1,
        max_masks_per_category: int = 40,
        use_mask_refinement: bool = False,
        merge_masks_per_class: bool = True,
        use_nms: bool = True,
    ) -> None:
        """Initialize the traceable SAM decoder."""
        super().__init__()
        self.predictor = sam_predictor
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_masks_per_category = max_masks_per_category
        self.use_mask_refinement = use_mask_refinement
        self.merge_masks_per_class = merge_masks_per_class
        self.use_nms = use_nms
        self.device = sam_predictor.device

    def _preprocess_points(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess points for SAM predictor.

        Args:
            points: Points tensor [max_points, 4] with (x, y, score, label)

        Returns:
            Tuple of (point_coords, point_labels, original_points)
        """
        valid_points = points[points[:, 3] != 0]

        # Check if there are any foreground points (label == 1)
        fg_mask = valid_points[:, 3] == 1

        # Transform coordinates
        coords = valid_points[:, :2]

        # Separate foreground and background
        bg_mask = valid_points[:, 3] == -1

        fg_coords = coords[fg_mask]
        bg_coords = coords[bg_mask]

        # Keep original points for output (contains x, y, score, label)
        fg_original = valid_points[fg_mask]
        bg_original = valid_points[bg_mask]

        num_fg = fg_coords.size(0)
        num_bg = bg_coords.size(0)

        # Pair each foreground with all background points
        bg_coords_expanded = bg_coords.unsqueeze(0).expand(num_fg, -1, -1)

        # Combine: [fg_point, bg_points...]
        point_coords = torch.cat([fg_coords.unsqueeze(1), bg_coords_expanded], dim=1)

        # Labels: 1 for fg, -1 for bg
        fg_labels = torch.ones(num_fg, 1, device=points.device, dtype=torch.float32)
        bg_labels = -torch.ones(num_fg, num_bg, device=points.device, dtype=torch.float32)
        point_labels = torch.cat([fg_labels, bg_labels], dim=1)

        # Combine original points: all foreground + all background
        original_points = torch.cat([fg_original, bg_original], dim=0)  # [num_fg + num_bg, 4]

        return point_coords, point_labels, original_points

    def _pad_outputs(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        label: int,
        original_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad outputs to fixed size.

        Args:
            masks: Masks [N, H, W]
            scores: Scores [N]
            label: Category label
            original_size: Original image size (H, W)

        Returns:
            Padded masks [max_masks, H, W], scores [max_masks], labels [max_masks]
        """
        device = self.device
        h, w = original_size

        num_masks = masks.size(0)
        max_masks = self.max_masks_per_category

        padded_masks = torch.zeros(max_masks, h, w, device=device, dtype=torch.bool)
        padded_scores = torch.zeros(max_masks, device=device, dtype=torch.float32)
        padded_labels = torch.full((max_masks,), -1, device=device, dtype=torch.int64)

        if torch.onnx.is_in_onnx_export():
            n = torch.minimum(num_masks, torch.tensor(max_masks))
        else:
            n = min(num_masks, max_masks)
        padded_masks[:n] = masks[:n]
        padded_scores[:n] = scores[:n]
        padded_labels[:n] = label

        return padded_masks, padded_scores, padded_labels

    def _resize_similarity(
        self,
        similarity: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Resize similarity map to target size.

        Args:
            similarity: Similarity map [feat_size, feat_size]
            target_size: Target size (H, W)

        Returns:
            Resized similarity [1, H, W]
        """
        sim = similarity.unsqueeze(0).unsqueeze(0)
        sim_resized = functional.interpolate(sim, size=target_size, mode="bilinear", align_corners=False)
        return sim_resized[0]

    def _predict_masks_for_category(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        similarity: torch.Tensor,
        original_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict and refine masks for a single category.

        Args:
            point_coords: Point coordinates [num_fg, num_points, 2]
            point_labels: Point labels [num_fg, num_points]
            similarity: Similarity map [feat_size, feat_size]
            original_size: Original image size (H, W)

        Returns:
            Tuple of (masks [N, H, W], scores [N])
        """
        # Initial prediction
        # masks: [num_fg, 1, H, W], iou_preds: [num_fg, 1]
        masks, iou_preds, low_res_logits = self.predictor.forward(
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=None,
            mask_input=None,
            multimask_output=True,
        )

        # Filter empty masks
        masks = masks[:, 0, :, :]  # [num_fg, H, W]
        keep = masks.sum(dim=(-1, -2)) > 0
        if not keep.any():
            return (
                torch.empty(0, *original_size, device=masks.device),
                torch.empty(0, device=masks.device),
            )

        masks = masks[keep]
        low_res_logits = low_res_logits[keep]
        point_coords = point_coords[keep]
        point_labels = point_labels[keep]

        # Compute boxes from masks for NMS (and optionally for 2nd SAM prediction)
        # The masks are in original image coordinates, so boxes are too.
        # The predictor will handle coordinate transformation internally.
        boxes = masks_to_boxes_traceable(masks)
        boxes = boxes.unsqueeze(1)

        # Optionally refine masks with 2nd SAM prediction using box prompts
        if self.use_mask_refinement:
            masks, mask_weights, _ = self.predictor.forward(
                point_coords=point_coords,
                point_labels=point_labels,
                boxes=boxes,
                mask_input=low_res_logits,
                multimask_output=True,  # Match SamDecoder behavior
            )
            masks = masks[:, 0, :, :]  # [N, H, W]
        else:
            mask_weights = iou_preds[keep]

        # NMS - NOTE: torchvision NMS requires float32 inputs
        if self.use_nms:
            nms_indices = nms(
                boxes[:, 0, :].to(torch.float32),
                mask_weights[:, 0].to(torch.float32),
                iou_threshold=self.nms_iou_threshold,
            )
            masks = masks[nms_indices]
            mask_weights = mask_weights[nms_indices]

        # Similarity-based scoring - always compute
        sim_resized = self._resize_similarity(similarity, original_size)
        mask_sum = (sim_resized * masks).sum(dim=(1, 2))
        mask_area = masks.sum(dim=(1, 2)) + 1e-6  # Avoid div by zero
        mask_scores = mask_sum / mask_area
        weighted_scores = (mask_scores * mask_weights.T)[0, :]

        # Apply threshold via masking, NOT early return
        keep = weighted_scores > self.confidence_threshold

        # Zero out scores for filtered masks instead of removing them
        weighted_scores = torch.where(keep, weighted_scores, torch.zeros_like(weighted_scores))

        # For merge: compute merged result (handles empty case gracefully)
        if self.merge_masks_per_class:
            # Mask out filtered masks before merging
            valid_masks = masks * keep.unsqueeze(-1).unsqueeze(-1)
            merged_mask = (valid_masks.sum(0) > 0).unsqueeze(0)  # [1, H, W]

            # For max score, use -inf for invalid to ensure they're not selected
            scores_for_max = torch.where(keep, weighted_scores, torch.full_like(weighted_scores, -float("inf")))
            merged_score = scores_for_max.max().unsqueeze(0)
            # If all filtered out, score becomes -inf, clamp to 0
            merged_score = torch.clamp(merged_score, min=0.0)
            return merged_mask, merged_score

        # For non-merge case: return with validity mask
        # Caller should filter by scores > 0
        return masks, weighted_scores

    def _process_single_image_with_points(
        self,
        image: torch.Tensor,
        point_prompts: torch.Tensor,
        similarities: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a single image with point prompts.

        Args:
            image: Input image [3, H, W]
            point_prompts: Point prompts [C, max_points, 4] with (x, y, score, label)
            similarities: Similarity maps [C, feat_size, feat_size]
            category_ids: Category ID mapping [C]

        Returns:
            pred_masks: [num_valid_masks, H, W]
            pred_scores: [num_valid_masks]
            pred_labels: [num_valid_masks]
            pred_points: [num_points_used, 4]
        """
        h, w = image.size(1), image.size(2)
        orig_size = (h, w)
        self.predictor.set_image(image)

        num_categories = len(category_ids)
        device = self.device

        all_masks = torch.zeros(num_categories, self.max_masks_per_category, h, w, device=device)
        all_scores = torch.zeros(num_categories, self.max_masks_per_category, device=device)
        all_labels = torch.full((num_categories, self.max_masks_per_category), -1, device=device, dtype=torch.int64)
        used_points_list: list[torch.Tensor] = []

        for class_idx in range(num_categories):
            class_id = category_ids[class_idx]
            points = point_prompts[class_idx]
            similarity = similarities[class_idx]

            point_coords, point_labels, original_points = self._preprocess_points(points)
            masks, scores = self._predict_masks_for_category(
                point_coords,
                point_labels,
                similarity,
                orig_size,
            )

            padded_masks, padded_scores, padded_labels = self._pad_outputs(
                masks,
                scores,
                class_id,
                orig_size,
            )

            all_masks[class_idx] = padded_masks
            all_scores[class_idx] = padded_scores
            all_labels[class_idx] = padded_labels

            if masks.numel() > 0 and masks.shape[0] > 0:
                used_points_list.append(original_points)
            else:
                used_points_list.append(torch.empty(0, 4, device=device))

        # Flatten and filter valid predictions
        valid_mask = all_labels >= 0
        pred_masks = all_masks.bool()[valid_mask]
        pred_scores = all_scores[valid_mask]
        pred_labels = all_labels[valid_mask]
        pred_points = torch.cat(used_points_list, dim=0)

        return pred_masks, pred_scores, pred_labels, pred_points

    def _process_single_image_with_boxes(
        self,
        image: torch.Tensor,
        box_prompts: torch.Tensor,
        category_ids: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a single image with box prompts.

        Args:
            image: Input image [3, H, W]
            box_prompts: Box prompts [C, max_boxes, 5] with (x1, y1, x2, y2, score)
            num_boxes: Number of valid boxes per category [C]
            category_ids: Category ID mapping [C]

        Returns:
            pred_masks: [num_valid_masks, H, W]
            pred_scores: [num_valid_masks]
            pred_labels: [num_valid_masks]
            pred_boxes: [num_valid_boxes, 5] with (x1, y1, x2, y2, score)
        """
        h, w = image.shape[1], image.shape[2]
        self.predictor.set_image(image)

        masks_list: list[torch.Tensor] = []
        scores_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        boxes_list: list[torch.Tensor] = []

        for class_idx, cat_id in enumerate(category_ids):
            n_valid = (box_prompts[class_idx][:, -1] != 0).sum().item()
            if n_valid == 0:
                continue
            boxes = box_prompts[class_idx, :n_valid]  # [N, 5]
            box_coords = boxes[:, :4]  # [N, 4]
            box_scores = boxes[:, 4]  # [N]

            # Predict masks for all boxes at once (batched)
            box_input = box_coords.unsqueeze(1)  # [N, 1, 4]

            masks, iou_preds, _ = self.predictor.forward(
                point_coords=None,
                point_labels=None,
                boxes=box_input,
                mask_input=None,
                multimask_output=False,
            )

            # masks shape: [N, 1, H, W], iou_preds shape: [N, 1]
            if masks.numel() > 0:
                masks = masks.squeeze(1)  # [N, H, W]
                iou_preds = iou_preds.squeeze(1)  # [N]

                # Filter out empty masks
                valid_mask = masks.sum(dim=(-1, -2)) > 0
                if valid_mask.any():
                    valid_masks = masks[valid_mask]
                    valid_iou = iou_preds[valid_mask]
                    valid_box_scores = box_scores[valid_mask]
                    valid_box_coords = box_coords[valid_mask]

                    combined_scores = (valid_iou + valid_box_scores) / 2

                    masks_list.append(valid_masks)
                    scores_list.append(combined_scores)
                    labels_list.append(
                        torch.full((valid_masks.shape[0],), cat_id, device=masks.device, dtype=torch.int64),
                    )
                    boxes_list.append(valid_box_coords)

        # Handle empty results
        if not masks_list:
            return (
                torch.empty(0, h, w, device=self.device, dtype=torch.bool),
                torch.empty(0, device=self.device),
                torch.empty(0, device=self.device, dtype=torch.int64),
                torch.empty(0, 5, device=self.device),
            )

        # Concatenate results (each element already has shape [N, ...])
        pred_masks = torch.cat(masks_list)
        pred_scores = torch.cat(scores_list)
        pred_labels = torch.cat(labels_list)
        pred_boxes = torch.cat(boxes_list)

        # Add scores to boxes for output [N, 5]
        pred_boxes_with_scores = torch.cat([pred_boxes, pred_scores.unsqueeze(1)], dim=1)

        return pred_masks, pred_scores, pred_labels, pred_boxes_with_scores

    def forward(
        self,
        images: list[torch.Tensor],
        category_ids: list[int],
        point_prompts: torch.Tensor | None = None,
        box_prompts: torch.Tensor | None = None,
        similarities: torch.Tensor | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Forward pass - predict masks from point or box prompts.

        Provide either (point_prompts, num_points, similarities) for point-based
        segmentation, or (box_prompts, num_boxes) for box-based segmentation.

        Args:
            images: List of input images, each [3, H, W]
            category_ids: Category ID mapping [C]
            point_prompts: Point prompts [T, C, max_points, 4] (optional)
            box_prompts: Box prompts [T, C, max_boxes, 5] (optional)
            similarities: Similarity maps [T, C, feat_size, feat_size] (optional)

        Returns:
            List of predictions per image, each containing:
                "pred_masks": [num_valid_masks, H, W]
                "pred_scores": [num_valid_masks]
                "pred_labels": [num_valid_masks]
                "pred_points": [num_points_used, 4] (point mode only)
                "pred_boxes": [num_boxes, 5] (box mode only)
        """
        use_points = point_prompts is not None
        use_boxes = box_prompts is not None

        if use_points == use_boxes:
            msg = "Provide either point_prompts or box_prompts, not both or neither"
            raise ValueError(msg)

        predictions: list[dict[str, torch.Tensor]] = []

        if use_points:
            for image, pts, sims in zip(images, point_prompts, similarities, strict=True):
                masks, scores, labels, points = self._process_single_image_with_points(
                    image,
                    pts,
                    sims,
                    category_ids,
                )
                predictions.append({
                    "pred_masks": masks,
                    "pred_scores": scores,
                    "pred_labels": labels,
                    "pred_points": points,
                })
        else:
            for image, bxs in zip(images, box_prompts, strict=True):
                masks, scores, labels, boxes = self._process_single_image_with_boxes(
                    image,
                    bxs,
                    category_ids,
                )
                predictions.append({
                    "pred_masks": masks,
                    "pred_scores": scores,
                    "pred_labels": labels,
                    "pred_boxes": boxes,
                })

        return predictions

    def forward_export(
        self,
        image: torch.Tensor,
        category_ids: torch.Tensor,
        point_prompts: torch.Tensor,
        similarities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Export-friendly forward for single image returning flat tensors.

        This method is designed for ONNX/TorchScript export where outputs
        must be tensors (not list[dict]).

        Args:
            image: Single input image [3, H, W]
            category_ids: Category ID mapping [C]
            point_prompts: Point prompts [C, max_points, 4] with (x, y, score, label)
            similarities: Similarity maps [C, feat_size, feat_size]

        Returns:
            Tuple of:
                masks: [num_valid_masks, H, W]
                scores: [num_valid_masks]
                labels: [num_valid_masks]
        """
        masks, scores, labels, _ = self._process_single_image_with_points(
            image,
            point_prompts,
            similarities,
            category_ids,
        )
        return masks, scores, labels
