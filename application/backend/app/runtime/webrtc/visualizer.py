# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import cv2
import numpy as np
import torch

from domain.services.schemas.processor import OutputData
from settings import get_settings

logger = logging.getLogger(__name__)


class InferenceVisualizer:
    """Visualizes inference results by overlaying masks and boxes on frames."""

    def __init__(self, enable_visualization: bool = True) -> None:
        """
        Initialize the visualizer.

        Args:
            enable_visualization: If False, returns frames unmodified.
        """
        self._enabled = enable_visualization
        settings = get_settings()
        self._mask_alpha = settings.mask_alpha
        self._mask_outline_thickness = settings.mask_outline_thickness
        self._box_thickness = settings.box_thickness
        self._font_scale = settings.label_font_scale
        self._font_thickness = settings.label_font_thickness

    def visualize(self, output_data: OutputData) -> np.ndarray:
        """
        Overlay inference results on the frame.

        Args:
            output_data: Output data containing frame, results, and label colors.
                frame: RGB frame in HWC format with dtype=uint8, shape (H, W, 3).
                results: List of predictions from model inference.
                    Each prediction contains:
                        "pred_masks": torch.Tensor of shape [num_masks, H, W]
                        "pred_points": torch.Tensor of shape [num_points, 4] with [x, y, score, fg_label]
                        "pred_boxes": torch.Tensor of shape [num_boxes, 5] with [x1, y1, x2, y2, score]
                        "pred_labels": torch.Tensor of shape [num_masks]
                label_colors: Mapping from label IDs (as strings) to RGB color tuples (0-255 range).
                    If None or label not found, generates deterministic color based on index.

        Returns:
            Annotated frame as numpy array in RGB HWC format (H, W, 3) with dtype=uint8.
            If visualization is disabled, returns the input frame unchanged.
        """
        if not self._enabled or not output_data.results:
            logger.debug("No inference results to visualize or visualization disabled.")
            return output_data.frame

        annotated = output_data.frame.copy()
        logger.info(f"Visualizing predictions {output_data.results}")

        for prediction in output_data.results:
            masks = prediction.get("pred_masks")
            boxes = prediction.get("pred_boxes")
            labels = prediction.get("pred_labels")

            if masks is not None:
                annotated = self._draw_masks(annotated, masks, labels, output_data.label_colors)

            if boxes is not None:
                annotated = self._draw_boxes(annotated, boxes, labels, output_data.label_colors)

        return annotated

    def _draw_masks(
        self,
        frame: np.ndarray,
        masks: torch.Tensor,
        labels: torch.Tensor | None,
        label_colors: dict[str, tuple[int, int, int]] | None,
    ) -> np.ndarray:
        """
        Draw segmentation masks on the frame.

        Args:
            frame: RGB frame in HWC format.
            masks: Tensor of shape [num_masks, H, W].
            labels: Tensor of shape [num_masks] with label IDs as strings.
            label_colors: Mapping from label ID to RGB color tuple.

        Returns:
            Frame with masks overlaid.
        """
        if masks.numel() == 0:
            return frame

        # Convert to numpy and ensure correct shape
        masks_np = masks.detach().cpu().numpy()

        overlay = frame.copy()

        for idx, mask in enumerate(masks_np):
            label_id = None
            if labels is not None and idx < len(labels):
                label_id = str(labels[idx].item())

            color = self._get_color(label_id, idx, label_colors)

            # Create boolean mask
            mask_bool = mask > 0.5

            # Apply semi-transparent overlay
            overlay[mask_bool] = (
                overlay[mask_bool] * (1 - self._mask_alpha) + np.array(color) * self._mask_alpha
            ).astype(np.uint8)

            # Draw thick outline around mask
            mask_uint8 = (mask_bool * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, self._mask_outline_thickness)

        return overlay

    def _draw_boxes(
        self,
        frame: np.ndarray,
        boxes: torch.Tensor,
        labels: torch.Tensor | None,
        label_colors: dict[str, tuple[int, int, int]] | None,
    ) -> np.ndarray:
        """
        Draw bounding boxes on the frame.

        Args:
            frame: RGB frame in HWC format.
            boxes: Tensor of shape [num_boxes, 5] with [x1, y1, x2, y2, score].
            labels: Tensor of shape [num_boxes] with label IDs as strings.
            label_colors: Mapping from label ID to RGB color tuple.

        Returns:
            Frame with boxes drawn.
        """
        if boxes.numel() == 0:
            return frame

        boxes_np = boxes.detach().cpu().numpy()

        for i, box in enumerate(boxes_np):
            x1, y1, x2, y2, score = box

            label_id = None
            if labels is not None and i < len(labels):
                label_id = str(labels[i].item())

            color = self._get_color(label_id, i, label_colors)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=self._box_thickness)

            # Draw confidence score
            label_text = f"{score:.2f}"
            label_size, _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, self._font_scale, self._font_thickness
            )
            label_bg_y1 = max(int(y1) - label_size[1] - 4, 0)
            cv2.rectangle(
                frame, (int(x1), label_bg_y1), (int(x1) + label_size[0], int(y1)), color, thickness=cv2.FILLED
            )
            cv2.putText(
                frame,
                label_text,
                (int(x1), int(y1) - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self._font_scale,
                (255, 255, 255),
                self._font_thickness,
            )

        return frame

    def _get_color(
        self, label_id: str | None, index: int, label_colors: dict[str, tuple[int, int, int]] | None
    ) -> tuple[int, int, int]:
        """
        Get color for visualization.

        Args:
            label_id: Label ID as string, or None
            index: Fallback index for color generation
            label_colors: Label-to-color mapping

        Returns:
            RGB color tuple (0-255 range)
        """
        if label_colors and label_id and label_id in label_colors:
            return label_colors[label_id]

        return self._generate_color(index)

    @staticmethod
    def _generate_color(index: int) -> tuple[int, int, int]:
        """
        Generate a distinct RGB color based on index.
        Args:
            index: Object index.
        Returns:
            RGB color tuple with values in range [0, 255].
        """
        hue = (index * 67) % 180  # Spread colors across hue spectrum
        hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
        return int(rgb[0]), int(rgb[1]), int(rgb[2])
