# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Visualization of predictions."""

import colorsys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import tv_tensors


class Visualizer:
    """The class exports the images for visualization.

    If an output file already exists, a number will be appended to the filename
    (e.g., image_1.png, image_2.png) to avoid overwriting.
    """

    def __init__(self, output_folder: str, class_map: dict[int, str]) -> None:
        """Initializes the visualization class.

        Args:
            output_folder: Directory to save visualization images
            class_map: Dictionary mapping class indices to category names
        """
        super().__init__()
        self.output_folder = output_folder
        self.color_map = self._setup_colors(class_map)

    def visualize(
        self,
        images: list[tv_tensors.Image],
        predictions: list[dict[str, torch.Tensor]],
        file_names: list[str],
    ) -> None:
        """This method exports the visualization images.

        Args:
            images: List of images to visualize
            predictions: List of predictions to visualize
            file_names: List of file names to visualize
        """
        for image, prediction, file_name in zip(
            images,
            predictions,
            file_names,
            strict=False,
        ):
            self._process_single_image(image, prediction, file_name)

    @staticmethod
    def _setup_colors(class_map: dict[int, str]) -> dict[int, list[int]]:
        """Setup colors for each category.

        Args:
            class_map: Dictionary mapping class indices to category names.

        Returns:
            Dictionary mapping class indices to colors
        """
        color_map = {}
        for class_id in class_map:
            rgb_float = colorsys.hsv_to_rgb(class_id / float(len(class_map)), 1.0, 1.0)
            color_map[class_id] = [int(x * 255) for x in rgb_float]
        return color_map

    def _process_single_image(
        self,
        image: tv_tensors.Image,
        prediction: dict[str, torch.Tensor],
        file_name: str,
    ) -> None:
        """Process a single image for visualization.

        Args:
            image: Image to visualize
            prediction: Prediction to visualize
            file_name: File name to visualize
        """
        pred_masks = prediction["pred_masks"]
        pred_points = prediction["pred_points"]
        pred_boxes = prediction["pred_boxes"]
        pred_labels = prediction["pred_labels"]
        image_np = image.permute(1, 2, 0).numpy()
        height, _ = image_np.shape[:2]

        output_path = Path(self.output_folder) / file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # If file exists, append a number to avoid overwriting
        if output_path.exists():
            # Find next available filename by appending a number
            stem = output_path.stem
            suffix = output_path.suffix
            parent = output_path.parent
            counter = 1
            while output_path.exists():
                new_name = f"{stem}_{counter}{suffix}"
                output_path = parent / new_name
                counter += 1

        image_vis = image_np.copy()
        if pred_masks is not None:
            # Draw each instance mask with the same class color and a border
            for pred_label, pred_mask in zip(pred_labels, pred_masks, strict=False):
                pred_label = pred_label.item()
                pred_mask = pred_mask.cpu().numpy()

                # Apply mask with more transparency
                masked_img = np.where(pred_mask[..., None], self.color_map[pred_label], image_vis)
                # Ensure both arrays have the same data type for cv2.addWeighted
                masked_img = masked_img.astype(np.uint8)
                image_vis = cv2.addWeighted(image_vis, 0.6, masked_img, 0.4, 0)

                # Add border to the mask
                mask_uint8 = pred_mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image_vis, contours, -1, (255, 255, 255), 1)

        # Draw points and confidence scores if provided
        if pred_points is not None:
            for pred_label, pred_point in zip(pred_labels, pred_points, strict=False):
                # Draw star marker
                pred_point = pred_point.float().cpu().numpy()
                x, y, score, fg_label = int(pred_point[0]), int(pred_point[1]), pred_point[2], int(pred_point[3])
                size = int(height / 100)  # Scale marker size with image
                cv2.drawMarker(
                    image_vis,
                    (x, y),
                    (255, 255, 255),
                    cv2.MARKER_STAR if fg_label == 1.0 else cv2.MARKER_SQUARE,
                    size,
                )

                # Add confidence score text
                confidence = float(score)
                cv2.putText(
                    image_vis,
                    f"{confidence:.2f}",
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    height / 100,
                    (255, 255, 255),
                    1,
                )

        # Draw boxes and confidence scores if provided
        if pred_boxes is not None:
            for pred_label, pred_box in zip(pred_labels, pred_boxes, strict=False):
                pred_label = pred_label.item()
                pred_box = pred_box.cpu().numpy()
                x1, y1, x2, y2, score = pred_box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(image_vis, (x1, y1), (x2, y2), color=self.color_map[pred_label], thickness=2)
                cv2.putText(
                    image_vis,
                    f"{score:.2f}",
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    height / 100,
                    (255, 255, 255),
                    1,
                )

        # Save visualization
        cv2.imwrite(output_path, cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
