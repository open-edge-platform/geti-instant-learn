# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import cv2
import numpy as np
import torch

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

    def visualize(self, frame: np.ndarray, results: list[dict[str, torch.Tensor]]) -> np.ndarray:
        """
        Overlay inference results on the frame.

        Args:
            frame: RGB frame in HWC format with dtype=uint8, shape (H, W, 3).
            results: List of predictions from model inference.
                Each prediction contains:
                    "pred_masks": torch.Tensor of shape [num_masks, H, W]
                    "pred_points": torch.Tensor of shape [num_points, 4] with [x, y, score, fg_label]
                    "pred_boxes": torch.Tensor of shape [num_boxes, 5] with [x1, y1, x2, y2, score]
                    "pred_labels": torch.Tensor of shape [num_masks]

        Returns:
            Annotated frame as numpy array in RGB HWC format (H, W, 3) with dtype=uint8.
            If visualization is disabled, returns the input frame unchanged.
        """
        if not self._enabled or not results:
            logger.debug("No inference results to visualize or visualization disabled.")
            return frame

        annotated = frame.copy()
        logger.info(f"Visualizing predictions {results}")

        for prediction in results:
            masks = prediction.get("pred_masks")
            boxes = prediction.get("pred_boxes")
            # labels = prediction.get("pred_labels")  # label_ids as tensor

            if masks is not None:
                annotated = self._draw_masks(annotated, masks)

            if boxes is not None:
                annotated = self._draw_boxes(annotated, boxes)

        return annotated

    def _draw_masks(self, frame: np.ndarray, masks: torch.Tensor) -> np.ndarray:
        """
        Draw segmentation masks on the frame.

        Args:
            frame: RGB frame in HWC format.
            masks: Tensor of shape [num_masks, H, W].

        Returns:
            Frame with masks overlaid.
        """
        if masks.numel() == 0:
            return frame

        # Convert to numpy and ensure correct shape
        masks_np = masks.detach().cpu().numpy()
        # labels_np = labels.detach().cpu().numpy() if labels is not None else None

        overlay = frame.copy()

        for idx, mask in enumerate(masks_np):
            # Get label for this mask
            # label_id = labels[idx].item() if idx < len(labels) else None

            # Get color - use default if label_colors is None or label not found
            # if label_colors and label_id is not None:
            #     color = label_colors.get(label_id)
            # else:
            #     color = None

            # if color is None:
            # Generate a deterministic color based on index if no color is provided
            np.random.seed(idx)
            color = tuple(np.random.randint(0, 255, 3).tolist())

            # Apply mask overlay
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = (overlay[mask_bool] * (1 - 0.5) + np.array(color) * 0.5).astype(np.uint8)

        return overlay

    def _draw_boxes(self, frame: np.ndarray, boxes: torch.Tensor) -> np.ndarray:
        """
        Draw bounding boxes on the frame.

        Args:
            frame: RGB frame in HWC format.
            boxes: Tensor of shape [num_boxes, 5] with [x1, y1, x2, y2, score].

        Returns:
            Frame with boxes drawn.
        """
        if boxes.numel() == 0:
            return frame

        boxes_np = boxes.detach().cpu().numpy()
        # labels_np = labels.detach().cpu().numpy() if labels is not None else None

        for i, box in enumerate(boxes_np):
            x1, y1, x2, y2, score = box
            # Get color from label_id or fallback
            # label_id = str(labels_np[i]) if labels_np is not None and i < len(labels_np) else None
            # color = label_colors.get(label_id) if label_id else None
            # if color is None:

            logger.info("GENERATING BOX color for box %d", i)
            color = self._generate_color(i)

            # Draw rectangle
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)

            # Draw confidence score
            label = f"{score:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_bg_y1 = max(int(y1) - label_size[1] - 4, 0)
            cv2.rectangle(
                frame, (int(x1), label_bg_y1), (int(x1) + label_size[0], int(y1)), color, thickness=cv2.FILLED
            )
            cv2.putText(frame, label, (int(x1), int(y1) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

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
