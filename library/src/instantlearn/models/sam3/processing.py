# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ONNX-traceable image preprocessor for SAM3 model.

This module provides an ONNX-compatible image preprocessing, all operations are pure PyTorch to
ensure ONNX traceability.
"""

import torch
from torch import nn
from torch.nn import functional

from .post_processing import PostProcessingConfig, apply_post_processing


class Sam3Preprocessor(nn.Module):
    """ONNX-traceable image preprocessor for SAM3.

    This preprocessor handles image resizing, padding, and normalization using
    only PyTorch operations, making it fully ONNX-traceable.

    Args:
        target_size: The target size for the longest dimension of the image.
                    Default: 1008 (standard SAM3 input size).

    Attributes:
        target_size: The target size for the longest dimension.
        mean: SAM3 normalization mean [0.5, 0.5, 0.5] (registered as buffer).
        std: SAM3 normalization standard deviation [0.5, 0.5, 0.5] (registered as buffer).

    Example:
        >>> import torch
        >>> preprocessor = Sam3Preprocessor(target_size=1008)
        >>> image = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8)
        >>> pixel_values, original_sizes = preprocessor(image)
        >>> pixel_values.shape
        torch.Size([1, 3, 1008, 1008])
    """

    def __init__(self, target_size: int = 1008) -> None:
        """Initialize the preprocessor.

        Args:
            target_size: The target size for the longest dimension. Default: 1008.
        """
        super().__init__()
        self.target_size = target_size

        # Register SAM3 normalization constants as buffers for ONNX compatibility
        # SAM3 uses mean=0.5, std=0.5 to map [0,1] to [-1,1]
        self.register_buffer(
            "mean",
            torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).reshape(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).reshape(1, 3, 1, 1),
        )

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> tuple[int, int]:
        """Compute the output size given input size and target long side length.

        Scales the image such that the longest dimension becomes long_side_length
        while maintaining aspect ratio.

        Args:
            oldh: Original image height.
            oldw: Original image width.
            long_side_length: The target length for the longest dimension.

        Returns:
            Tuple of (new_height, new_width) maintaining aspect ratio.

        Example:
            >>> Sam3Preprocessor.get_preprocess_shape(480, 640, 1008)
            (756, 1008)
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh = int(oldh * scale + 0.5)
        neww = int(oldw * scale + 0.5)
        return (newh, neww)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocess image for SAM3 inference.

        Handles input format conversion, resizing to target size (without maintaining
        aspect ratio), and SAM3 normalization.

        Args:
            pixel_values: Input image tensor with shape (B, C, H, W).
                         Can be uint8 (0-255) or float (0-1).

        Raises:
            ValueError: If input tensor does not have 4 dimensions or if the number of channels is not 3.

        Returns:
            Tuple containing:
                - pixel_values: Preprocessed image tensor with shape (B, 3, target_size, target_size)
                  and SAM3 normalized values (range [-1, 1]).
                - original_sizes: Tensor with shape (B, 2) containing [height, width] of input images.

        Example:
            >>> preprocessor = Sam3Preprocessor(target_size=1008)
            >>> # uint8 input
            >>> img_uint8 = torch.randint(0, 256, (2, 3, 480, 640), dtype=torch.uint8)
            >>> pixel_values, orig_sizes = preprocessor(img_uint8)
            >>> pixel_values.shape
            torch.Size([2, 3, 1008, 1008])
            >>> orig_sizes.shape
            torch.Size([2, 2])
            >>> # float input
            >>> img_float = torch.rand(2, 3, 480, 640, dtype=torch.float32)
            >>> pixel_values, orig_sizes = preprocessor(img_float)
        """
        if pixel_values.ndim != 4:
            msg = f"Expected input shape (B, C, H, W), got {pixel_values.shape}"
            raise ValueError(msg)

        # Get original sizes before any processing
        batch_size = pixel_values.shape[0]
        original_height = pixel_values.shape[2]
        original_width = pixel_values.shape[3]
        original_sizes = torch.tensor(
            [[original_height, original_width]] * batch_size,
            dtype=torch.int32,
            device=pixel_values.device,
        )

        # Resize first while still uint8 (matches HF behavior)
        pixel_values = functional.interpolate(
            pixel_values.float(),  # F.interpolate needs float, but we'll round back
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        # Round to simulate uint8 behavior during resize, then rescale
        pixel_values = pixel_values.round().clamp(0, 255) / 255.0

        # Normalize with SAM3 mean and std (maps [0,1] to [-1,1])
        pixel_values = (pixel_values - self.mean) / self.std

        return pixel_values, original_sizes


class Sam3PromptPreprocessor(nn.Module):
    """ONNX-traceable prompt preprocessor for SAM3.

    This preprocessor handles box normalization and padding for SAM3 prompt inputs.
    It converts absolute box coordinates to normalized cxcywh format suitable for
    ONNX tracing.

    Args:
        target_size: The target size for the preprocessed image. Default: 1008.
        pad_value: Sentinel value used for padding boxes. Default: -10.0.

    Attributes:
        target_size: The target size for the preprocessed image.
        pad_value: Sentinel value for padding boxes.

    Example:
        >>> import torch
        >>> preprocessor = Sam3PromptPreprocessor(target_size=1008)
        >>> boxes = torch.tensor([[[100, 100, 200, 200]]], dtype=torch.float32)
        >>> img_sizes = torch.tensor([[480, 640]], dtype=torch.int32)
        >>> norm_boxes = preprocessor(boxes, img_sizes)
        >>> norm_boxes.shape
        torch.Size([1, 1, 4])
    """

    def __init__(self, target_size: int = 1008, pad_value: float = -10.0) -> None:
        """Initialize the prompt preprocessor.

        Args:
            target_size: The target size for the preprocessed image. Default: 1008.
            pad_value: Sentinel value used for padding boxes. Default: -10.0.
        """
        super().__init__()
        self.target_size = target_size
        self.pad_value = pad_value

    @staticmethod
    def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
        """Convert bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h) format.

        Args:
            boxes: Boxes in (x1, y1, x2, y2) format.

        Returns:
            Boxes in (cx, cy, w, h) format.
        """
        x0, y0, x1, y1 = boxes.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)

    def forward(
        self,
        input_boxes: torch.Tensor | list | tuple,
        original_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """Preprocess prompts for SAM3 inference.

        Normalizes boxes from absolute coordinates to [0, 1] range and converts
        from xyxy to cxcywh format.

        Args:
            input_boxes: Input boxes in xyxy format. Can be:
                - Tensor with shape (B, N, 4)
                - Tensor with shape (4,) for a single box
                - List/tuple [x1, y1, x2, y2] for a single box
                Coordinates are in absolute pixel values.
            original_sizes: Tensor with shape (B, 2) containing [height, width]
                           of the original images.

        Returns:
            Normalized boxes tensor with shape (B, N, 4) in cxcywh format,
            with values in [0, 1] range.

        Example:
            >>> preprocessor = Sam3PromptPreprocessor()
            >>> # Tensor input
            >>> boxes = torch.tensor([[[100, 100, 200, 200]]], dtype=torch.float32)
            >>> sizes = torch.tensor([[480, 640]], dtype=torch.int32)
            >>> norm_boxes = preprocessor(boxes, sizes)
            >>> # List input (single box)
            >>> norm_boxes = preprocessor([100, 100, 200, 200], sizes)
        """
        # Convert to tensor if needed and ensure shape (B, N, 4)
        if not isinstance(input_boxes, torch.Tensor):
            input_boxes = torch.tensor(input_boxes, dtype=torch.float32)
        input_boxes = input_boxes.to(device=original_sizes.device, dtype=torch.float32)

        if input_boxes.ndim == 1:  # (4,) -> (1, 1, 4)
            input_boxes = input_boxes.unsqueeze(0).unsqueeze(0)
        elif input_boxes.ndim == 2:  # (N, 4) -> (1, N, 4)
            input_boxes = input_boxes.unsqueeze(0)

        # Extract height and width from original_sizes (B, 2)
        heights = original_sizes[:, 0:1].float()  # (B, 1)
        widths = original_sizes[:, 1:2].float()  # (B, 1)

        # Create scale factor: [width, height, width, height]
        scale_factor = torch.cat([widths, heights, widths, heights], dim=1)  # (B, 4)
        scale_factor = scale_factor.unsqueeze(1)  # (B, 1, 4)

        # Normalize boxes to [0, 1] range
        normalized_boxes = input_boxes / scale_factor

        # Convert from xyxy to cxcywh format
        return self.box_xyxy_to_cxcywh(normalized_boxes)


class Sam3Postprocessor(nn.Module):
    """ONNX-traceable postprocessor for SAM3.

    This postprocessor handles the conversion of raw model outputs to final predictions
    with separate forward paths for ONNX (tensor-only) and eager mode (list-based).

    Args:
        target_size: The target size for mask interpolation. Default: 1008.

    Attributes:
        target_size: The target size for mask interpolation.

    Example:
        >>> import torch
        >>> postprocessor = Sam3Postprocessor(target_size=1008)
        >>> outputs = {
        ...     "pred_logits": torch.randn(1, 10),
        ...     "pred_boxes": torch.rand(1, 10, 4),
        ...     "pred_masks": torch.randn(1, 10, 256, 256),
        ...     "presence_logits": torch.randn(1, 1),
        ... }
        >>> target_sizes = [(480, 640)]
        >>> results = postprocessor.forward_eager(outputs, target_sizes)
    """

    def __init__(
        self,
        target_size: int = 1008,
        threshold: float = 0.3,
        mask_threshold: float = 0.5,
        post_processing: PostProcessingConfig | None = None,
    ) -> None:
        """Initialize the postprocessor.

        Args:
            target_size: The target size for mask interpolation. Default: 1008.
            threshold: Score threshold for filtering predictions. Default: 0.3.
            mask_threshold: Threshold for binarizing masks. Default: 0.5.
            post_processing: Optional post-processing configuration for NMS,
                mask overlap removal, and non-overlapping pixel constraints.
                When ``None``, no post-processing is applied beyond score
                thresholding and mask binarization.
        """
        super().__init__()
        self.target_size = target_size
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.post_processing = post_processing

    @staticmethod
    def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Convert bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.

        Args:
            boxes: Boxes in (cx, cy, w, h) format.

        Returns:
            Boxes in (x1, y1, x2, y2) format.
        """
        x_c, y_c, w, h = boxes.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    @staticmethod
    def _scale_boxes(boxes: torch.Tensor, target_sizes: list[tuple[int, int]]) -> torch.Tensor:
        """Scale batch of bounding boxes to target sizes.

        Args:
            boxes: Bounding boxes of shape (batch_size, num_boxes, 4).
                  Each box is expected to be in (x1, y1, x2, y2) format.
            target_sizes: Target sizes to scale to. Each target size is expected
                         to be in (height, width) format.

        Returns:
            Scaled bounding boxes of shape (batch_size, num_boxes, 4).
        """
        image_height = torch.tensor([i[0] for i in target_sizes])
        image_width = torch.tensor([i[1] for i in target_sizes])

        scale_factor = torch.stack([image_width, image_height, image_width, image_height], dim=1)
        scale_factor = scale_factor.unsqueeze(1).to(boxes.device)
        boxes *= scale_factor
        return boxes

    def _preprocess_outputs(
        self,
        outputs: dict,
        target_sizes: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract outputs and post-process model predictions.

        Computes scores with presence weighting, applies sigmoid to masks,
        and scales boxes to target sizes.

        Args:
            outputs: Dictionary containing model outputs.
            target_sizes: List of (height, width) tuples for each image.

        Returns:
            Tuple of (batch_scores, batch_boxes, batch_masks) tensors.
        """
        pred_logits = outputs["pred_logits"]  # (B, N)
        pred_boxes = outputs["pred_boxes"]  # (B, N, 4) normalized [0,1]
        pred_masks = outputs["pred_masks"]  # (B, N, H, W)
        presence_logits = outputs.get("presence_logits")  # (B, 1) or None

        # Compute scores with optional presence weighting
        batch_scores = pred_logits.sigmoid()
        if presence_logits is not None:
            batch_scores *= presence_logits.sigmoid()

        # Apply sigmoid to mask logits
        batch_masks = pred_masks.sigmoid()

        # Scale boxes to target sizes
        batch_boxes = self._scale_boxes(pred_boxes, target_sizes)

        return batch_scores, batch_boxes, batch_masks

    def _postprocess_single(
        self,
        scores: torch.Tensor,
        boxes: torch.Tensor,
        masks: torch.Tensor,
        target_size: tuple[int, int],
        threshold: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Postprocess predictions for a single image.

        Filters by score threshold, interpolates kept masks to target size,
        and binarizes masks.

        Args:
            scores: Confidence scores (N,).
            boxes: Bounding boxes (N, 4).
            masks: Mask logits after sigmoid (N, H, W).
            target_size: Target (height, width) for mask interpolation.
            threshold: Score threshold override. Uses ``self.threshold``
                when None. Default: None.

        Returns:
            Tuple of (kept_scores, kept_boxes, kept_masks) with only predictions
            above the threshold.
        """
        # Filter by score threshold
        effective_threshold = threshold if threshold is not None else self.threshold
        keep = scores > effective_threshold
        kept_scores = scores[keep]
        kept_boxes = boxes[keep]
        kept_masks = masks[keep]  # (num_keep, H, W)

        # Interpolate kept masks to target size
        if kept_masks.numel() > 0:
            kept_masks = functional.interpolate(
                kept_masks.unsqueeze(0),  # (1, num_keep, H, W)
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # (num_keep, H, W)

        # Binarize masks
        kept_masks = (kept_masks > self.mask_threshold).to(torch.int64)

        # Apply configurable post-processing (NMS, IoM suppression, etc.)
        kept_scores, kept_boxes, kept_masks = apply_post_processing(
            kept_scores,
            kept_boxes,
            kept_masks,
            self.post_processing,
        )

        return kept_scores, kept_boxes, kept_masks

    @staticmethod
    def _filter_by_points(
        scores: torch.Tensor,
        boxes: torch.Tensor,
        masks: torch.Tensor,
        input_points: torch.Tensor,
        input_points_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter predictions to keep only masks that contain input points.

        When using the DETR grounding path for point prompts, detection scores
        are text-based (dot-product with text features). The highest-scored mask
        may not overlap with the input point(s). This method filters to keep
        only masks that actually contain the positive input points, effectively
        emulating the SAM-style interactive segmentation behavior.

        Args:
            scores: Confidence scores (N,).
            boxes: Bounding boxes (N, 4) in pixel coordinates.
            masks: Binary masks (N, H, W) at target image resolution.
            input_points: Point coordinates (P, 2) as (x, y) in pixel space.
            input_points_labels: Point labels (P,) where 1=positive, 0=negative.
                If None, all points are treated as positive.

        Returns:
            Filtered (scores, boxes, masks) containing only predictions whose
            masks intersect with the positive input points. Falls back to
            original predictions if no masks match.
        """
        if masks.numel() == 0 or input_points.numel() == 0:
            return scores, boxes, masks

        if input_points_labels is not None:
            positive_mask = input_points_labels == 1
            positive_points = input_points[positive_mask]
        else:
            positive_points = input_points

        if positive_points.numel() == 0:
            return scores, boxes, masks

        h, w = masks.shape[-2:]
        num_masks = masks.shape[0]
        num_pos = positive_points.shape[0]

        # Check which masks contain each positive point
        # point_hits[i, j] = True if mask i contains positive point j
        point_hits = torch.zeros(num_masks, num_pos, dtype=torch.bool, device=masks.device)
        for j in range(num_pos):
            px = int(positive_points[j, 0].clamp(0, w - 1).item())
            py = int(positive_points[j, 1].clamp(0, h - 1).item())
            point_hits[:, j] = masks[:, py, px] > 0

        # Prefer masks that contain ALL positive points
        all_points_hit = point_hits.all(dim=1)
        if all_points_hit.any():
            return scores[all_points_hit], boxes[all_points_hit], masks[all_points_hit]

        # Fall back to masks that contain at least one positive point
        any_point_hit = point_hits.any(dim=1)
        if any_point_hit.any():
            return scores[any_point_hit], boxes[any_point_hit], masks[any_point_hit]

        # No masks contain any point — return all (unchanged behavior)
        return scores, boxes, masks

    def forward(
        self,
        outputs: dict,
        target_sizes: torch.Tensor | list[tuple[int, int]],
        input_points: list[torch.Tensor] | None = None,
        input_points_labels: list[torch.Tensor] | None = None,
    ) -> list[dict[str, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Postprocess model outputs to final predictions.

        Args:
            outputs: Dictionary containing model outputs.
            target_sizes: Target sizes as tensor (B, 2) or list of (height, width) tuples.
            input_points: Per-image point prompts as list of (P, 2) tensors in
                pixel coordinates. When provided, masks are filtered to keep only
                those intersecting with positive input points. Default: None.
            input_points_labels: Per-image point labels as list of (P,) tensors
                where 1=positive, 0=negative. Used together with input_points.
                Default: None.

        Returns:
            ONNX mode: Tuple of (scores, boxes, masks) tensors (filtered).
            Eager mode: List of dicts with 'scores', 'boxes', 'masks' keys.
        """
        batch_scores, batch_boxes, batch_masks = self._preprocess_outputs(outputs, target_sizes)

        results = []
        for idx, (scores, boxes, masks) in enumerate(zip(batch_scores, batch_boxes, batch_masks, strict=False)):
            has_points = input_points is not None and idx < len(input_points) and input_points[idx].numel() > 0

            if has_points:
                # When point prompts are provided, use a much lower threshold
                # to capture candidate masks that contain the input points. The
                # correct mask may score low in text-based scoring but still
                # intersect with the prompted point(s).
                point_threshold = min(self.threshold, 0.01)
                kept_scores, kept_boxes, kept_masks = self._postprocess_single(
                    scores,
                    boxes,
                    masks,
                    target_sizes[idx],
                    threshold=point_threshold,
                )

                pts = input_points[idx]
                pts_labels = input_points_labels[idx] if input_points_labels is not None else None
                kept_scores, kept_boxes, kept_masks = self._filter_by_points(
                    kept_scores,
                    kept_boxes,
                    kept_masks,
                    pts,
                    pts_labels,
                )
            else:
                kept_scores, kept_boxes, kept_masks = self._postprocess_single(
                    scores,
                    boxes,
                    masks,
                    target_sizes[idx],
                )

            if torch.onnx.is_in_onnx_export():
                results.append((kept_scores, kept_boxes, kept_masks))
            else:
                results.append({"scores": kept_scores, "boxes": kept_boxes, "masks": kept_masks})

        return results
