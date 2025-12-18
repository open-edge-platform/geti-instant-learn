# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Traceable SAM decoder for ONNX/TorchScript export."""

from logging import getLogger

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import masks_to_boxes, nms

from getiprompt.components.sam.pytorch import PyTorchSAMPredictor
from getiprompt.data import ResizeLongestSide

logger = getLogger("Geti Prompt")

__all__ = ["TraceableSamDecoder"]


class TraceableSamDecoder(nn.Module):
    """Traceable SAM decoder that accepts tensor inputs for ONNX/TorchScript export.

    This decoder processes point prompts and similarities as tensors instead of
    dictionaries, enabling full pipeline traceability.

    Args:
        sam_predictor: PyTorch SAM predictor instance.
        target_length: Target length for image preprocessing. Default: 1024.
        mask_similarity_threshold: Threshold for similarity-based mask filtering. Default: 0.38.
        nms_iou_threshold: IoU threshold for NMS. Default: 0.1.
        max_masks_per_category: Maximum masks to return per category (for padding). Default: 10.
    """

    def __init__(
        self,
        sam_predictor: PyTorchSAMPredictor,
        target_length: int = 1024,
        mask_similarity_threshold: float = 0.38,
        nms_iou_threshold: float = 0.1,
        max_masks_per_category: int = 40,
    ) -> None:
        """Initialize the traceable SAM decoder."""
        super().__init__()
        self.predictor = sam_predictor
        self.mask_similarity_threshold = mask_similarity_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_masks_per_category = max_masks_per_category
        self.transform = ResizeLongestSide(target_length)
        self.device = sam_predictor.device

    def _preprocess_image(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Preprocess image for SAM.

        Args:
            image: Image tensor [3, H, W]

        Returns:
            Preprocessed image and original size
        """
        original_size = (image.shape[-2], image.shape[-1])
        preprocessed = self.transform.apply_image_torch(image).to(self.device)
        return preprocessed, original_size

    def _preprocess_points(
        self,
        points: torch.Tensor,
        num_valid: int,
        original_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Preprocess points for SAM predictor.

        Args:
            points: Points tensor [max_points, 4] with (x, y, score, label)
            num_valid: Number of valid points
            original_size: Original image size (H, W)

        Returns:
            Tuple of (point_coords, point_labels, original_points) or None if no valid foreground
        """
        if num_valid == 0:
            return None

        valid_points = points[:num_valid]

        # Check if there are any foreground points (label == 1)
        fg_mask = valid_points[:, 3] == 1
        if not fg_mask.any():
            return None

        # Transform coordinates
        coords = valid_points[:, :2]
        transformed_coords = self.transform.apply_coords_torch(coords, original_size)

        # Separate foreground and background
        bg_mask = valid_points[:, 3] == -1

        fg_coords = transformed_coords[fg_mask]
        bg_coords = transformed_coords[bg_mask]

        # Keep original points for output (contains x, y, score, label)
        fg_original = valid_points[fg_mask]
        bg_original = valid_points[bg_mask]

        num_fg = fg_coords.shape[0]
        num_bg = bg_coords.shape[0]

        if num_bg == 0:
            point_coords = fg_coords.unsqueeze(1)  # [num_fg, 1, 2]
            point_labels = torch.ones(num_fg, 1, device=points.device, dtype=torch.float32)
            original_points = fg_original  # [num_fg, 4]
            return point_coords, point_labels, original_points

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
        sim_resized = F.interpolate(sim, size=target_size, mode="bilinear", align_corners=False)
        return sim_resized.squeeze(0)

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
        masks, iou_preds, low_res_logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=None,
            mask_input=None,
            multimask_output=True,
        )

        # Filter empty masks
        keep = masks.squeeze(1).sum(dim=(-1, -2)) > 0
        if not keep.any():
            return (
                torch.empty(0, *original_size, device=masks.device),
                torch.empty(0, device=masks.device),
            )

        masks = masks[keep]
        low_res_logits = low_res_logits[keep]
        point_coords = point_coords[keep]
        point_labels = point_labels[keep]

        # Refine with boxes
        boxes = masks_to_boxes(masks.squeeze(1))
        boxes = self.transform.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        boxes = boxes.reshape(-1, 4).unsqueeze(1)

        masks, mask_weights, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=boxes,
            mask_input=low_res_logits,
            multimask_output=False,
        )

        # NMS
        nms_indices = nms(
            boxes.squeeze(1),
            mask_weights.squeeze(1),
            iou_threshold=self.nms_iou_threshold,
        )

        masks = masks[nms_indices].squeeze(1)
        mask_weights = mask_weights[nms_indices]

        # Similarity-based filtering
        if similarity.numel() > 0:
            sim_resized = self._resize_similarity(similarity, original_size)
            mask_sum = (sim_resized * masks).sum(dim=(1, 2))
            mask_area = masks.sum(dim=(1, 2))
            mask_scores = mask_sum / (mask_area + 1e-6)
            weighted_scores = mask_scores * mask_weights.squeeze(1)
            keep = weighted_scores > self.mask_similarity_threshold

            if not keep.any():
                return (
                    torch.empty(0, *original_size, device=masks.device),
                    torch.empty(0, device=masks.device),
                )

            masks = masks[keep]
            weighted_scores = weighted_scores[keep]
        else:
            weighted_scores = mask_weights.squeeze(1)

        return masks, weighted_scores

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
        device = masks.device if masks.numel() > 0 else self.device
        dtype = masks.dtype if masks.numel() > 0 else torch.float32
        h, w = original_size

        num_masks = masks.shape[0] if masks.numel() > 0 else 0
        max_masks = self.max_masks_per_category

        padded_masks = torch.zeros(max_masks, h, w, device=device, dtype=dtype)
        padded_scores = torch.zeros(max_masks, device=device, dtype=dtype)
        padded_labels = torch.full((max_masks,), -1, device=device, dtype=torch.int64)

        if num_masks > 0:
            n = min(num_masks, max_masks)
            padded_masks[:n] = masks[:n]
            padded_scores[:n] = scores[:n]
            padded_labels[:n] = label

        return padded_masks, padded_scores, padded_labels

    def _process_single_image(
        self,
        image: torch.Tensor,
        point_prompts: torch.Tensor,
        num_points: torch.Tensor,
        similarities: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Process a single image with all categories.

        Args:
            image: Input image [3, H, W]
            point_prompts: Point prompts [C, max_points, 4]
            num_points: Number of valid points per category [C]
            similarities: Similarity maps [C, feat_size, feat_size]
            category_ids: Category ID mapping [C]

        Returns:
            masks: [C, max_masks, H, W]
            scores: [C, max_masks]
            labels: [C, max_masks]
            num_masks: [C]
            used_points: List of points used per category [N_c, 4]
        """
        preprocessed_image, orig_size = self._preprocess_image(image)
        h, w = orig_size

        self.predictor.set_image(preprocessed_image, orig_size)

        num_categories = category_ids.size(0)
        device = image.device

        all_masks = torch.zeros(num_categories, self.max_masks_per_category, h, w, device=device)
        all_scores = torch.zeros(num_categories, self.max_masks_per_category, device=device)
        all_labels = torch.full((num_categories, self.max_masks_per_category), -1, device=device, dtype=torch.int64)
        num_masks_out = torch.zeros(num_categories, device=device, dtype=torch.int64)
        used_points_per_category: list[torch.Tensor] = []

        for class_idx in range(num_categories):
            class_id = category_ids[class_idx].item()
            points = point_prompts[class_idx]
            n_valid = num_points[class_idx].item()
            similarity = similarities[class_idx]

            result = self._preprocess_points(points, n_valid, orig_size)
            if result is None:
                used_points_per_category.append(torch.empty(0, 4, device=device))
                continue

            point_coords, point_labels, original_points = result

            masks, scores = self._predict_masks_for_category(
                point_coords,
                point_labels,
                similarity,
                orig_size,
            )

            padded_masks, padded_scores, padded_labels = self._pad_outputs(
                masks, scores, class_id, orig_size
            )

            all_masks[class_idx] = padded_masks
            all_scores[class_idx] = padded_scores
            all_labels[class_idx] = padded_labels
            num_masks_out[class_idx] = min(masks.shape[0] if masks.numel() > 0 else 0, self.max_masks_per_category)

            # Only include points if we got masks
            if masks.numel() > 0 and masks.shape[0] > 0:
                used_points_per_category.append(original_points)
            else:
                used_points_per_category.append(torch.empty(0, 4, device=device))

        return all_masks.bool(), all_scores, all_labels, num_masks_out, used_points_per_category

    def forward(
        self,
        images: list[torch.Tensor],
        point_prompts: torch.Tensor,
        num_points: torch.Tensor,
        similarities: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Forward pass - predict masks from point prompts for multiple images.

        Args:
            images: List of input images, each [3, H, W]
            point_prompts: Point prompts [T, C, max_points, 4]
            num_points: Number of valid points [T, C]
            similarities: Similarity maps [T, C, feat_size, feat_size]
            category_ids: Category ID mapping [C]

        Returns:
            List of predictions per image, each containing:
                "pred_masks": [num_valid_masks, H, W]
                "pred_scores": [num_valid_masks]
                "pred_labels": [num_valid_masks]
                "pred_points": [num_points_used, 4] with (x, y, score, label)
        """
        predictions: list[dict[str, torch.Tensor]] = []

        for image, prompts, n_points, sims in zip(
            images, 
            point_prompts,
            num_points,
            similarities,
            strict=True,
        ):
            masks, scores, labels, _, used_points = self._process_single_image(
                image,
                prompts,
                n_points,
                sims,
                category_ids,
            )

            # Filter valid masks (label >= 0)
            valid_mask = labels >= 0

            # Concatenate all used points across categories
            all_points = torch.cat(used_points, dim=0) if used_points else torch.empty(0, 4, device=masks.device)

            predictions.append({
                "pred_masks": masks[valid_mask],
                "pred_scores": scores[valid_mask],
                "pred_labels": labels[valid_mask],
                "pred_points": all_points,
            })

        return predictions
