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
        assert pixel_values.ndim == 4, f"Expected BCHW tensor, got shape {pixel_values.shape}"

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
        normalized_boxes = self.box_xyxy_to_cxcywh(normalized_boxes)

        return normalized_boxes


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

    def __init__(self, target_size: int = 1008) -> None:
        """Initialize the postprocessor.

        Args:
            target_size: The target size for mask interpolation. Default: 1008.
        """
        super().__init__()
        self.target_size = target_size

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

    def forward(
        self,
        outputs: dict,
        target_sizes: torch.Tensor | list[tuple[int, int]],
        threshold: float = 0.3,
        mask_threshold: float = 0.5,
    ) -> list[dict[str, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Postprocess model outputs, dispatching to ONNX or eager mode.

        Automatically selects the appropriate implementation based on whether
        ONNX export is in progress.

        Args:
            outputs: Dictionary containing model outputs.
            target_sizes: Target sizes as tensor (B, 2) or list of (height, width) tuples.
            threshold: Score threshold for predictions. Default: 0.3.
            mask_threshold: Threshold for binarizing masks. Default: 0.5.

        Returns:
            ONNX mode: Tuple of (scores, boxes, masks, counts) tensors.
            Eager mode: List of dicts with 'scores', 'boxes', 'masks' keys.
        """
        if torch.onnx.is_in_onnx_export():
            if not isinstance(target_sizes, torch.Tensor):
                msg = "ONNX mode requires tensor target_sizes"
                raise TypeError(msg)
            return self.forward_onnx(outputs, target_sizes, threshold, mask_threshold)

        # Convert tensor to list for eager mode
        if isinstance(target_sizes, torch.Tensor):
            target_sizes = [(int(target_sizes[i, 0]), int(target_sizes[i, 1])) for i in range(target_sizes.shape[0])]
        return self.forward_eager(outputs, target_sizes, threshold, mask_threshold)

    def forward_onnx(
        self,
        outputs: dict,
        target_sizes: torch.Tensor,
        threshold: float = 0.3,
        mask_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Postprocess model outputs for ONNX inference.

        ONNX-compatible version that returns padded tensors instead of Python lists.

        Args:
            outputs: Dictionary containing model outputs with keys:
                    - pred_logits: (B, N) confidence scores
                    - pred_boxes: (B, N, 4) boxes in xyxy format (normalized [0,1])
                    - pred_masks: (B, N, H, W) mask logits
                    - presence_logits: (B, 1) presence scores (optional)
            target_sizes: Tensor with shape (B, 2) containing [height, width] for each image.
            threshold: Score threshold to keep instance predictions. Default: 0.3.
            mask_threshold: Threshold for binarizing the predicted masks. Default: 0.5.

        Returns:
            Tuple of (scores, boxes, masks, counts):
                - scores: Padded scores tensor (B, max_detections)
                - boxes: Padded boxes tensor (B, max_detections, 4)
                - masks: Padded masks tensor (B, max_detections, H, W)
                - counts: Number of valid detections per image (B,)

        Example:
            >>> postprocessor = Sam3Postprocessor()
            >>> outputs = {...}  # Model outputs
            >>> target_sizes = torch.tensor([[480, 640]], dtype=torch.int32)
            >>> scores, boxes, masks, counts = postprocessor.forward_onnx(outputs, target_sizes)
        """
        # Extract outputs
        pred_logits = outputs["pred_logits"]  # (B, N)
        pred_boxes = outputs["pred_boxes"]  # (B, N, 4) in xyxy format (normalized [0,1])
        pred_masks = outputs["pred_masks"]  # (B, N, H, W)
        presence_logits = outputs.get("presence_logits")  # (B, 1) or None

        # Compute scores
        batch_scores = pred_logits.sigmoid()
        if presence_logits is not None:
            presence_scores = presence_logits.sigmoid()  # (B, 1)
            batch_scores = batch_scores * presence_scores  # Broadcast multiplication

        # Apply sigmoid to mask logits
        batch_masks = pred_masks.sigmoid()

        # Boxes are already in xyxy format from the model
        batch_boxes = pred_boxes

        # Scale boxes to target sizes
        # Convert target_sizes tensor to list of tuples for compatibility
        batch_size = target_sizes.shape[0]
        target_sizes_list = [(target_sizes[i, 0].item(), target_sizes[i, 1].item()) for i in range(batch_size)]
        batch_boxes = self._scale_boxes(batch_boxes, target_sizes_list)

        # Interpolate masks to target size
        all_masks = []
        for idx in range(batch_size):
            target_h, target_w = target_sizes[idx, 0].item(), target_sizes[idx, 1].item()
            masks = batch_masks[idx : idx + 1]  # (1, N, H, W)
            masks = functional.interpolate(
                masks,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            all_masks.append(masks)
        batch_masks = torch.cat(all_masks, dim=0)  # (B, N, target_H, target_W)

        # Binarize masks
        batch_masks = (batch_masks > mask_threshold).float()

        # Filter by score threshold
        keep_mask = batch_scores > threshold  # (B, N)

        # For ONNX compatibility, we return padded tensors with counts
        # instead of variable-length lists
        return batch_scores, batch_boxes, batch_masks, keep_mask.sum(dim=1)

    def forward_eager(
        self,
        outputs: dict,
        target_sizes: list[tuple[int, int]],
        threshold: float = 0.3,
        mask_threshold: float = 0.5,
    ) -> list[dict[str, torch.Tensor]]:
        """Postprocess model outputs for eager mode inference.

        Returns Python list of dicts for compatibility with current API.

        Args:
            outputs: Dictionary containing model outputs with keys:
                    - pred_logits: (B, N) confidence scores
                    - pred_boxes: (B, N, 4) boxes in xyxy format (normalized [0,1])
                    - pred_masks: (B, N, H, W) mask logits
                    - presence_logits: (B, 1) presence scores (optional)
            target_sizes: List of tuples containing (height, width) for each image.
            threshold: Score threshold to keep instance predictions. Default: 0.3.
            mask_threshold: Threshold for binarizing the predicted masks. Default: 0.5.

        Returns:
            List of dictionaries, each containing:
                - scores: Confidence scores for kept instances (N_keep,)
                - boxes: Bounding boxes in xyxy format (N_keep, 4)
                - masks: Binary segmentation masks (N_keep, H, W)

        Example:
            >>> postprocessor = Sam3Postprocessor()
            >>> outputs = {...}  # Model outputs
            >>> target_sizes = [(480, 640)]
            >>> results = postprocessor.forward_eager(outputs, target_sizes)
            >>> results[0]["scores"]  # Scores for first image
        """
        # Extract outputs
        pred_logits = outputs["pred_logits"]  # (B, N)
        pred_boxes = outputs["pred_boxes"]  # (B, N, 4) in xyxy format (normalized [0,1])
        pred_masks = outputs["pred_masks"]  # (B, N, H, W)
        presence_logits = outputs.get("presence_logits")  # (B, 1) or None

        batch_size = pred_logits.shape[0]  # noqa: F841

        # Compute scores
        batch_scores = pred_logits.sigmoid()
        if presence_logits is not None:
            presence_scores = presence_logits.sigmoid()  # (B, 1)
            batch_scores = batch_scores * presence_scores  # Broadcast multiplication

        # Apply sigmoid to mask logits
        batch_masks = pred_masks.sigmoid()

        # Boxes are already in xyxy format from the model
        batch_boxes = pred_boxes

        # Scale boxes to target sizes
        batch_boxes = self._scale_boxes(batch_boxes, target_sizes)

        # Process each image in the batch
        results = []
        for idx, (score_vec, box_vec, mask_vec) in enumerate(zip(batch_scores, batch_boxes, batch_masks, strict=False)):
            # Filter by score threshold
            keep = score_vec > threshold
            kept_scores = score_vec[keep]
            kept_boxes = box_vec[keep]
            kept_masks = mask_vec[keep]  # (num_keep, H, W)

            # Resize masks to target size
            target_size = target_sizes[idx]
            if len(kept_masks) > 0:
                kept_masks = functional.interpolate(
                    kept_masks.unsqueeze(0),  # (1, num_keep, H, W)
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)  # (num_keep, target_H, target_W)

            # Binarize masks
            kept_masks = (kept_masks > mask_threshold).long()

            results.append({"scores": kept_scores, "boxes": kept_boxes, "masks": kept_masks})

        return results
