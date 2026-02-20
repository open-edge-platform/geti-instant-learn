# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM3 OpenVINO inference examples.

This script demonstrates SAM3OpenVINO usage with the same images and prompts
from the project README and SAM3 notebook. It covers:

    1. Text prompting via fit() — detect elephants across multiple images
    2. Per-sample text prompting — no fit() required
    3. Multi-category text prompting — detect multiple object types at once
    4. Box prompting — segment specific regions of interest
    5. Combined text + box prompting — both prompt types together

Usage:
    # Using local OpenVINO model directory
    python examples/sam3_openvino_example.py --model-dir ./sam3-openvino/openvino-fp16

    # Download from HuggingFace Hub
    python examples/sam3_openvino_example.py --repo-id rajeshgangireddy/sam3_openvino

    # With visualization saved to disk
    python examples/sam3_openvino_example.py --model-dir ./sam3-openvino/openvino-fp16 --save-viz

Note:
    SAM3 v2 ONNX models do NOT support point prompts natively. Points must be
    converted to small bounding boxes (see Example 5 for a workaround).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download

from instantlearn.data import Sample
from instantlearn.data.utils import read_image
from instantlearn.models import SAM3OpenVINO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
logger = logging.getLogger(__name__)

# Paths to COCO example images (relative to library/ directory)
COCO_ASSETS = Path("examples/assets/coco")
IMAGE_ELEPHANT_1 = COCO_ASSETS / "000000286874.jpg"  # Elephant
IMAGE_ELEPHANT_2 = COCO_ASSETS / "000000173279.jpg"  # Elephant herd
IMAGE_ELEPHANT_3 = COCO_ASSETS / "000000390341.jpg"  # Elephant
IMAGE_ELEPHANT_4 = COCO_ASSETS / "000000267704.jpg"  # Elephant with person


def resolve_model_dir(args: argparse.Namespace) -> Path:
    """Resolve model directory from CLI args — local path or HuggingFace download.

    Args:
        args: Parsed CLI arguments with model_dir and repo_id.

    Returns:
        Path to the directory containing OpenVINO IR models.

    Raises:
        FileNotFoundError: If the specified model directory does not exist.
        ValueError: If neither --model-dir nor --repo-id is provided.
    """
    if args.model_dir:
        model_dir = Path(args.model_dir)
        if not model_dir.exists():
            msg = f"Model directory not found: {model_dir}"
            raise FileNotFoundError(msg)
        return model_dir

    if args.repo_id:
        logger.info("Downloading models from HuggingFace: %s", args.repo_id)
        local_dir = snapshot_download(repo_id=args.repo_id)
        return Path(local_dir)

    msg = "Provide either --model-dir or --repo-id"
    raise ValueError(msg)


def print_prediction_summary(
    prediction: dict[str, torch.Tensor],
    *,
    categories: list[str] | None = None,
) -> None:
    """Print a compact summary of a single image prediction.

    Args:
        prediction: Dictionary with pred_masks, pred_boxes, pred_labels.
        categories: Optional category name list for label-to-name mapping.
    """
    n_masks = len(prediction["pred_masks"])
    if n_masks == 0:
        logger.info("  No objects detected.")
        return

    logger.info("  Found %d object(s)", n_masks)

    # Show per-detection info
    for i in range(n_masks):
        box = prediction["pred_boxes"][i]
        label_id = prediction["pred_labels"][i].item()
        score = box[4].item() if box.shape[0] == 5 else 0.0
        coords = box[:4].tolist()

        label_str = f"id={label_id}"
        if categories and 0 <= label_id < len(categories):
            label_str = f"{categories[label_id]} (id={label_id})"

        logger.info(
            "    [%d] %s  score=%.3f  box=[%.0f, %.0f, %.0f, %.0f]  mask=%s",
            i,
            label_str,
            score,
            *coords,
            tuple(prediction["pred_masks"][i].shape),
        )


def save_visualization(
    image_path: Path,
    prediction: dict[str, torch.Tensor],
    output_path: Path,
    *,
    categories: list[str] | None = None,
) -> None:
    """Save a simple overlay visualization of predictions on the image.

    Args:
        image_path: Path to the original image.
        prediction: Prediction dictionary with pred_masks, pred_boxes, pred_labels.
        output_path: Path to save the visualization.
        categories: Optional list of category names.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning("Could not read image: %s", image_path)
        return

    # Colors for different labels (BGR)
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
    ]

    overlay = image.copy()
    for i in range(len(prediction["pred_masks"])):
        mask = prediction["pred_masks"][i].numpy()
        box = prediction["pred_boxes"][i][:4].int().tolist()
        label_id = prediction["pred_labels"][i].item()
        score = prediction["pred_boxes"][i][4].item() if prediction["pred_boxes"][i].shape[0] == 5 else 0.0
        color = colors[label_id % len(colors)]

        # Draw mask overlay
        if mask.shape[:2] == image.shape[:2]:
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = (
                np.array(overlay[mask_bool], dtype=np.float32) * 0.5 + np.array(color, dtype=np.float32) * 0.5
            ).astype(np.uint8)

        # Draw bounding box
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Draw label text
        label = f"{label_id}"
        if categories and 0 <= label_id < len(categories):
            label = categories[label_id]
        text = f"{label}: {score:.2f}"
        cv2.putText(overlay, text, (box[0], max(box[1] - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    logger.info("  Visualization saved: %s", output_path)


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


def example_1_text_prompt_with_fit(model: SAM3OpenVINO, *, save_viz: bool = False) -> None:
    """Example 1: Text prompting via fit() — same as README.

    Mirrors the README example: fit once with category="elephant", then predict
    on multiple images without specifying categories again.
    """
    logger.info("=" * 70)
    logger.info("Example 1: Text Prompting via fit()")
    logger.info("=" * 70)

    # fit() stores categories so predict() reuses them for every image
    ref_sample = Sample(categories=["elephant"], category_ids=[0])
    model.fit(ref_sample)

    # Predict on multiple images (same as README)
    targets = [
        Sample(image_path=str(IMAGE_ELEPHANT_1)),
        Sample(image_path=str(IMAGE_ELEPHANT_2)),
    ]

    t0 = time.perf_counter()
    predictions = model.predict(targets)
    elapsed = time.perf_counter() - t0
    logger.info("Inference on %d images took %.2f s (%.2f s/image)", len(targets), elapsed, elapsed / len(targets))

    for idx, (target, pred) in enumerate(zip(targets, predictions, strict=True)):
        logger.info("Image %d: %s", idx, Path(target.image_path).name)
        print_prediction_summary(pred, categories=["elephant"])
        if save_viz:
            save_visualization(
                Path(target.image_path),
                pred,
                Path(f"outputs/sam3_ov_ex1_img{idx}.jpg"),
                categories=["elephant"],
            )

    # Reset fit state
    model.category_mapping = None


def example_2_per_sample_text_prompt(model: SAM3OpenVINO, *, save_viz: bool = False) -> None:
    """Example 2: Per-sample text prompting (no fit required).

    Each sample carries its own categories — useful when different images
    need different prompts.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("Example 2: Per-Sample Text Prompting (no fit)")
    logger.info("=" * 70)

    targets = [
        Sample(
            image_path=str(IMAGE_ELEPHANT_3),
            categories=["elephant"],
            category_ids=[0],
        ),
        Sample(
            image_path=str(IMAGE_ELEPHANT_4),
            categories=["elephant"],
            category_ids=[0],
        ),
    ]

    t0 = time.perf_counter()
    predictions = model.predict(targets)
    elapsed = time.perf_counter() - t0
    logger.info("Inference: %.2f s", elapsed)

    for idx, (target, pred) in enumerate(zip(targets, predictions, strict=True)):
        logger.info("Image %d: %s", idx, Path(target.image_path).name)
        print_prediction_summary(pred, categories=["elephant"])
        if save_viz:
            save_visualization(
                Path(target.image_path),
                pred,
                Path(f"outputs/sam3_ov_ex2_img{idx}.jpg"),
                categories=["elephant"],
            )


def example_3_multi_category(model: SAM3OpenVINO, *, save_viz: bool = False) -> None:
    """Example 3: Multi-category text prompting.

    Detect multiple object types in a single image. The image with elephants
    and people is a good candidate.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("Example 3: Multi-Category Text Prompting")
    logger.info("=" * 70)

    categories = ["elephant", "person", "tree"]
    category_ids = [0, 1, 2]

    target = Sample(
        image_path=str(IMAGE_ELEPHANT_4),
        categories=categories,
        category_ids=category_ids,
    )

    t0 = time.perf_counter()
    predictions = model.predict(target)
    elapsed = time.perf_counter() - t0
    logger.info("Inference: %.2f s", elapsed)

    logger.info("Image: %s", IMAGE_ELEPHANT_4.name)
    print_prediction_summary(predictions[0], categories=categories)
    if save_viz:
        save_visualization(
            IMAGE_ELEPHANT_4,
            predictions[0],
            Path("outputs/sam3_ov_ex3_multi_category.jpg"),
            categories=categories,
        )


def example_4_box_prompt(model: SAM3OpenVINO, *, save_viz: bool = False) -> None:
    """Example 4: Box prompting — segment a specific region.

    Provide a bounding box in xyxy format to segment the object within that region.
    This mirrors the box prompt example from the SAM3 notebook.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("Example 4: Box Prompting")
    logger.info("=" * 70)

    image = read_image(str(IMAGE_ELEPHANT_1))
    _, h, w = image.shape
    logger.info("Image size: %d x %d", w, h)

    # Place a box roughly around the main elephant
    # For 000000286874.jpg (elephant), a reasonable box covering the central elephant
    box_xyxy = [150, 100, 500, 400]
    logger.info("Box prompt (xyxy): %s", box_xyxy)

    target = Sample(
        image=image,
        bboxes=torch.tensor([box_xyxy]),
    )

    t0 = time.perf_counter()
    predictions = model.predict(target)
    elapsed = time.perf_counter() - t0
    logger.info("Inference: %.2f s", elapsed)

    print_prediction_summary(predictions[0])
    if save_viz:
        save_visualization(
            IMAGE_ELEPHANT_1,
            predictions[0],
            Path("outputs/sam3_ov_ex4_box_prompt.jpg"),
        )


def example_5_point_as_box(model: SAM3OpenVINO, *, save_viz: bool = False) -> None:
    """Example 5: Point prompt workaround using a small box.

    SAM3 v2 ONNX models do not have a dedicated point prompt input.
    A common workaround is to convert a point click into a small bounding box
    centered on that point. This simulates point prompting behavior.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("Example 5: Point Prompt (via small box workaround)")
    logger.info("=" * 70)

    image = read_image(str(IMAGE_ELEPHANT_1))
    _, h, w = image.shape

    # Simulate a point click at the center of the elephant
    point_x, point_y = 320, 260
    # Convert point to a small box (±margin pixels)
    margin = 10
    box_from_point = [
        max(0, point_x - margin),
        max(0, point_y - margin),
        min(w, point_x + margin),
        min(h, point_y + margin),
    ]
    logger.info("Point click: (%d, %d) → box: %s", point_x, point_y, box_from_point)

    target = Sample(
        image=image,
        bboxes=torch.tensor([box_from_point]),
    )

    t0 = time.perf_counter()
    predictions = model.predict(target)
    elapsed = time.perf_counter() - t0
    logger.info("Inference: %.2f s", elapsed)

    print_prediction_summary(predictions[0])
    if save_viz:
        save_visualization(
            IMAGE_ELEPHANT_1,
            predictions[0],
            Path("outputs/sam3_ov_ex5_point_as_box.jpg"),
        )


def example_6_combined_text_and_box(model: SAM3OpenVINO, *, save_viz: bool = False) -> None:
    """Example 6: Combined text + box prompting.

    Provide both a text category and a bounding box. The text guides what to
    segment, and the box constrains where to look.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("Example 6: Combined Text + Box Prompting")
    logger.info("=" * 70)

    image = read_image(str(IMAGE_ELEPHANT_4))

    # Provide both a category and a box
    target = Sample(
        image=image,
        categories=["elephant"],
        category_ids=[0],
        bboxes=torch.tensor([[100, 80, 450, 380]]),
    )

    t0 = time.perf_counter()
    predictions = model.predict(target)
    elapsed = time.perf_counter() - t0
    logger.info("Inference: %.2f s", elapsed)

    print_prediction_summary(predictions[0], categories=["elephant"])
    if save_viz:
        save_visualization(
            IMAGE_ELEPHANT_4,
            predictions[0],
            Path("outputs/sam3_ov_ex6_combined.jpg"),
            categories=["elephant"],
        )


def main() -> None:
    """Run all SAM3 OpenVINO examples."""
    parser = argparse.ArgumentParser(
        description="SAM3 OpenVINO inference examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Local directory containing OpenVINO IR or ONNX model files.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace repo ID to download models from (e.g., rajeshgangireddy/sam3_openvino).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="OpenVINO device: CPU, GPU, AUTO (default: CPU).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5).",
    )
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save visualizations to outputs/ directory.",
    )
    parser.add_argument(
        "--examples",
        type=str,
        default="all",
        help="Comma-separated example numbers to run, e.g. '1,3,4' (default: all).",
    )
    args = parser.parse_args()

    # Resolve model directory
    model_dir = resolve_model_dir(args)
    logger.info("Model directory: %s", model_dir)

    # Initialize SAM3 OpenVINO model
    t0 = time.perf_counter()
    model = SAM3OpenVINO(
        model_dir=model_dir,
        device=args.device,
        confidence_threshold=args.confidence,
    )
    logger.info("Model loaded in %.2f s", time.perf_counter() - t0)

    # Map of example functions
    examples = {
        1: example_1_text_prompt_with_fit,
        2: example_2_per_sample_text_prompt,
        3: example_3_multi_category,
        4: example_4_box_prompt,
        5: example_5_point_as_box,
        6: example_6_combined_text_and_box,
    }

    # Determine which examples to run
    selected = list(examples.keys()) if args.examples == "all" else [int(x.strip()) for x in args.examples.split(",")]

    save_viz = args.save_viz
    for num in selected:
        if num in examples:
            examples[num](model, save_viz=save_viz)
        else:
            logger.warning("Unknown example number: %d (available: %s)", num, list(examples.keys()))

    logger.info("")
    logger.info("Done! All examples completed.")


if __name__ == "__main__":
    main()
