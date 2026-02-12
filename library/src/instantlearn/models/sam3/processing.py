# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ONNX-traceable image preprocessor for SAM3 model.

This module provides an ONNX-compatible image preprocessing, all operations are pure PyTorch to
ensure ONNX traceability.
"""

import torch
from torch import nn
from torch.nn import functional


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

    This preprocessor handles box and point normalization for SAM3 prompt inputs.
    It converts absolute coordinates to normalized format suitable for ONNX tracing:
    - Boxes: absolute xyxy -> normalized cxcywh [0, 1]
    - Points: absolute xy -> normalized xy [0, 1]

    Args:
        target_size: The target size for the preprocessed image. Default: 1008.
        pad_value: Sentinel value used for padding boxes. Default: -10.0.

    Attributes:
        target_size: The target size for the preprocessed image.
        pad_value: Sentinel value for padding boxes.

    Example:
        >>> import torch
        >>> preprocessor = Sam3PromptPreprocessor(target_size=1008)
        >>> sizes = torch.tensor([[480, 640]], dtype=torch.int32)
        >>> # Box only
        >>> boxes, points = preprocessor(sizes, input_boxes=[100, 100, 200, 200])
        >>> boxes.shape
        torch.Size([1, 1, 4])
        >>> # Point only
        >>> boxes, points = preprocessor(sizes, input_points=[150, 150])
        >>> points.shape
        torch.Size([1, 1, 2])
        >>> # Both boxes and points
        >>> boxes, points = preprocessor(
        ...     sizes,
        ...     input_boxes=torch.tensor([[[100, 100, 200, 200]]]),
        ...     input_points=torch.tensor([[[150, 150]]]),
        ... )
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

    def _normalize_boxes(
        self,
        input_boxes: torch.Tensor | list | tuple,
        original_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize boxes from absolute xyxy to normalized cxcywh format."""
        if not isinstance(input_boxes, torch.Tensor):
            input_boxes = torch.tensor(input_boxes, dtype=torch.float32)
        input_boxes = input_boxes.to(device=original_sizes.device, dtype=torch.float32)

        if input_boxes.ndim == 1:  # (4,) -> (1, 1, 4)
            input_boxes = input_boxes.unsqueeze(0).unsqueeze(0)
        elif input_boxes.ndim == 2:  # (N, 4) -> (1, N, 4)
            input_boxes = input_boxes.unsqueeze(0)

        heights = original_sizes[:, 0:1].float()
        widths = original_sizes[:, 1:2].float()
        scale_factor = torch.cat([widths, heights, widths, heights], dim=1).unsqueeze(1)
        normalized_boxes = input_boxes / scale_factor
        return self.box_xyxy_to_cxcywh(normalized_boxes)

    def _normalize_points(
        self,
        input_points: torch.Tensor | list | tuple,
        original_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize points from absolute xy to normalized [0, 1] xy format."""
        if not isinstance(input_points, torch.Tensor):
            input_points = torch.tensor(input_points, dtype=torch.float32)
        input_points = input_points.to(device=original_sizes.device, dtype=torch.float32)

        if input_points.ndim == 1:  # (2,) -> (1, 1, 2)
            input_points = input_points.unsqueeze(0).unsqueeze(0)
        elif input_points.ndim == 2:  # (N, 2) -> (1, N, 2)
            input_points = input_points.unsqueeze(0)

        heights = original_sizes[:, 0:1].float()
        widths = original_sizes[:, 1:2].float()
        scale_factor = torch.cat([widths, heights], dim=1).unsqueeze(1)
        return input_points / scale_factor

    def forward(
        self,
        original_sizes: torch.Tensor,
        input_boxes: torch.Tensor | list | tuple | None = None,
        input_points: torch.Tensor | list | tuple | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Preprocess box and point prompts for SAM3 inference.

        Normalizes coordinates from absolute pixel values to [0, 1] range:
        - Boxes: xyxy -> cxcywh normalized
        - Points: xy -> xy normalized

        Args:
            original_sizes: Tensor with shape (B, 2) containing [height, width]
                           of the original images.
            input_boxes: Input boxes in xyxy format. Can be:
                - Tensor with shape (B, N, 4)
                - Tensor with shape (4,) for a single box
                - List/tuple [x1, y1, x2, y2] for a single box
                - None if no box prompts
            input_points: Input points in xy format. Can be:
                - Tensor with shape (B, N, 2)
                - Tensor with shape (2,) for a single point
                - List/tuple [x, y] for a single point
                - None if no point prompts

        Returns:
            Tuple of (normalized_boxes, normalized_points) where:
                - normalized_boxes: (B, N, 4) in cxcywh format or None
                - normalized_points: (B, N, 2) in xy format or None

        Example:
            >>> preprocessor = Sam3PromptPreprocessor()
            >>> sizes = torch.tensor([[480, 640]], dtype=torch.int32)
            >>> # Box only
            >>> boxes, points = preprocessor(sizes, input_boxes=[100, 100, 200, 200])
            >>> # Point only
            >>> boxes, points = preprocessor(sizes, input_points=[150, 200])
            >>> # Both
            >>> boxes, points = preprocessor(
            ...     sizes,
            ...     input_boxes=torch.tensor([[[100, 100, 200, 200]]]),
            ...     input_points=torch.tensor([[[150, 200]]]),
            ... )
        """
        normalized_boxes = None
        normalized_points = None

        if input_boxes is not None:
            normalized_boxes = self._normalize_boxes(input_boxes, original_sizes)

        if input_points is not None:
            normalized_points = self._normalize_points(input_points, original_sizes)

        return normalized_boxes, normalized_points


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
    ) -> None:
        """Initialize the postprocessor.

        Args:
            target_size: The target size for mask interpolation. Default: 1008.
            threshold: Score threshold for filtering predictions. Default: 0.3.
            mask_threshold: Threshold for binarizing masks. Default: 0.5.
        """
        super().__init__()
        self.target_size = target_size
        self.threshold = threshold
        self.mask_threshold = mask_threshold

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Postprocess predictions for a single image.

        Filters by score threshold, interpolates kept masks to target size,
        and binarizes masks.

        Args:
            scores: Confidence scores (N,).
            boxes: Bounding boxes (N, 4).
            masks: Mask logits after sigmoid (N, H, W).
            target_size: Target (height, width) for mask interpolation.

        Returns:
            Tuple of (kept_scores, kept_boxes, kept_masks) with only predictions
            above the threshold.
        """
        # Filter by score threshold
        keep = scores > self.threshold
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

        return kept_scores, kept_boxes, kept_masks

    def forward(
        self,
        outputs: dict,
        target_sizes: torch.Tensor | list[tuple[int, int]],
    ) -> list[dict[str, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Postprocess model outputs to final predictions.

        Args:
            outputs: Dictionary containing model outputs.
            target_sizes: Target sizes as tensor (B, 2) or list of (height, width) tuples.

        Returns:
            ONNX mode: Tuple of (scores, boxes, masks) tensors (filtered).
            Eager mode: List of dicts with 'scores', 'boxes', 'masks' keys.
        """
        batch_scores, batch_boxes, batch_masks = self._preprocess_outputs(outputs, target_sizes)

        results = []
        for idx, (scores, boxes, masks) in enumerate(zip(batch_scores, batch_boxes, batch_masks, strict=False)):
            kept_scores, kept_boxes, kept_masks = self._postprocess_single(scores, boxes, masks, target_sizes[idx])
            if torch.onnx.is_in_onnx_export():
                # In ONNX export, we return tensors with all predictions (including those below threshold)
                results.append((kept_scores, kept_boxes, kept_masks))
            else:
                results.append({"scores": kept_scores, "boxes": kept_boxes, "masks": kept_masks})

        return results
