# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Visualization of predictions."""

import colorsys
from pathlib import Path

import cv2
import numpy as np

from instantlearn.data.base.prediction import Prediction


def setup_colors(class_map: dict[int, str]) -> dict[int, list[int]]:
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


def render_predictions(
    image_rgb: np.ndarray,
    prediction: Prediction,
    color_map: dict[int, list[int]],
    *,
    show_scores: bool = True,
) -> np.ndarray:
    """Render prediction overlays (masks, boxes) onto an RGB image.

    This is the shared rendering function used by both ``visualize_single_image``
    (which additionally saves to disk) and external callers such as web UIs. It
    operates directly on the numpy :class:`~instantlearn.data.base.prediction.Prediction`
    contract (masks ``(N, H, W)``, ``label_ids`` ``(N,)``, ``boxes`` ``(N, 4)``
    xyxy, ``scores`` ``(N,)``).

    Args:
        image_rgb: Image in RGB uint8 format (H, W, 3).
        prediction: A ``Prediction`` with ``masks`` / ``label_ids`` and optional
            ``boxes`` (xyxy) and ``scores``.
        color_map: Mapping from class id to ``[R, G, B]`` color.
        show_scores: Whether to draw confidence scores next to boxes.

    Returns:
        Annotated RGB image (uint8).
    """
    pred_masks = prediction.masks if prediction.masks is not None else np.empty((0,))
    pred_labels = prediction.label_ids if prediction.label_ids is not None else np.empty((0,), dtype=np.int32)
    pred_boxes = prediction.boxes if prediction.boxes is not None else np.empty((0, 4))
    pred_scores = prediction.scores if prediction.scores is not None else np.empty((0,))

    image_vis = image_rgb.copy()
    h, w = image_vis.shape[:2]

    # Draw masks
    if len(pred_masks):
        for pred_label, pred_mask in zip(pred_labels, pred_masks, strict=False):
            pred_label = int(pred_label)
            pred_mask = np.asarray(pred_mask)

            # Resize mask to image dimensions if needed
            if pred_mask.shape != (h, w):
                mask_uint8 = pred_mask.astype(np.uint8) * 255
                mask_uint8 = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
                pred_mask = mask_uint8 > 0

            color = color_map.get(pred_label, [0, 255, 0])
            masked_img = np.where(pred_mask[..., None], color, image_vis)
            masked_img = masked_img.astype(np.uint8)
            image_vis = cv2.addWeighted(image_vis, 0.6, masked_img, 0.4, 0)

            mask_uint8 = pred_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_vis, contours, -1, (255, 255, 255), 1)

    # Draw boxes
    if len(pred_boxes):
        for idx, (pred_label, pred_box) in enumerate(zip(pred_labels, pred_boxes, strict=False)):
            pred_label = int(pred_label)
            x1, y1, x2, y2 = (int(v) for v in np.asarray(pred_box, dtype=np.float32)[:4])
            color = color_map.get(pred_label, [0, 255, 0])
            cv2.rectangle(image_vis, (x1, y1), (x2, y2), color=color, thickness=2)
            if show_scores and idx < len(pred_scores):
                label = f"{100 * float(pred_scores[idx]):.0f}%"
                cv2.putText(image_vis, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image_vis


def visualize_single_image(
    image: np.ndarray,
    prediction: Prediction,
    file_name: str,
    output_folder: str,
    color_map: dict[int, list[int]],
) -> np.ndarray:
    """Render predictions onto an image and save the result to disk.

    Delegates rendering to ``render_predictions`` and handles file output
    with automatic deduplication of existing filenames.

    Args:
        image: Image to visualize as an RGB uint8 numpy array (H, W, 3).
        prediction: A ``Prediction`` (see ``render_predictions``).
        file_name: Output file name.
        output_folder: Directory to save visualization images.
        color_map: Mapping from class id to ``[R, G, B]`` color.

    Returns:
        Annotated RGB image (uint8).
    """
    image_np = np.asarray(image)

    output_path = Path(output_folder) / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Avoid overwriting existing files
    if output_path.exists():
        stem = output_path.stem
        suffix = output_path.suffix
        parent = output_path.parent
        counter = 1
        while output_path.exists():
            new_name = f"{stem}_{counter}{suffix}"
            output_path = parent / new_name
            counter += 1

    image_vis = render_predictions(image_np, prediction, color_map)

    cv2.imwrite(str(output_path), cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
    return image_vis


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
        self.color_map = setup_colors(class_map)

    def visualize(
        self,
        images: list[np.ndarray],
        predictions: list[Prediction],
        file_names: list[str],
    ) -> None:
        """This method exports the visualization images.

        Args:
            images: List of RGB uint8 images (H, W, 3) to visualize
            predictions: List of ``Prediction`` objects to visualize
            file_names: List of file names to visualize
        """
        for image, prediction, file_name in zip(
            images,
            predictions,
            file_names,
            strict=False,
        ):
            visualize_single_image(
                image,
                prediction,
                file_name,
                self.output_folder,
                self.color_map,
            )
