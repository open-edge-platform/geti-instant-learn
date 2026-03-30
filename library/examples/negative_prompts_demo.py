# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Negative prompts demo — reduce false positives with background masks.

This script demonstrates how to use negative masks (background annotations)
to suppress false positive detections in the Matcher model.

The workflow:
    1. Provide a reference image with a foreground mask (object of interest)
       and a background mask (region to exclude).
    2. The library extracts negative point prompts from the background mask.
    3. During inference, negative points guide SAM to avoid the excluded region.

Usage:
    cd library
    python examples/negative_prompts_demo.py

    # With a custom reference image and interactive mask painting:
    python examples/negative_prompts_demo.py --interactive

    # With a specific model:
    python examples/negative_prompts_demo.py --model per_dino
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from instantlearn.data import Sample
from instantlearn.data.base.sample import BACKGROUND_CATEGORY, BACKGROUND_CATEGORY_ID
from instantlearn.models import Matcher
from instantlearn.visualizer import render_predictions, setup_colors

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR / "assets"
OUTPUT_DIR = SCRIPT_DIR / "output" / "negative_prompts"


def create_synthetic_background_mask(
    image_shape: tuple[int, int],
    region: tuple[int, int, int, int],
) -> np.ndarray:
    """Create a binary background mask for a rectangular region.

    Args:
        image_shape: (H, W) of the image.
        region: (y1, x1, y2, x2) bounding box of the background region.

    Returns:
        Binary mask (H, W) with 1 inside the region.
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    y1, x1, y2, x2 = region
    mask[y1:y2, x1:x2] = 1
    return mask


def run_comparison(
    ref_image_path: str,
    ref_mask_path: str,
    target_image_paths: list[str],
    background_region: tuple[int, int, int, int] | None = None,
    background_mask: np.ndarray | None = None,
    device: str = "cuda",
    num_negative_points: int = 5,
) -> dict[str, list[dict]]:
    """Run prediction with and without negative prompts and compare results.

    Args:
        ref_image_path: Path to the reference image.
        ref_mask_path: Path to the reference foreground mask.
        target_image_paths: Paths to target images.
        background_region: Optional (y1, x1, y2, x2) region to mark as background.
        background_mask: Optional pre-made background mask (H, W).
        device: Device for inference.
        num_negative_points: Number of negative points per background mask.

    Returns:
        Dict with "without_negative" and "with_negative" prediction lists.
    """
    results = {}

    # --- Run WITHOUT negative prompts (baseline) ---
    logger.info("Running Matcher WITHOUT negative prompts (baseline)...")
    model_baseline = Matcher(device=device, num_negative_points=num_negative_points)

    ref_sample_baseline = Sample(
        image_path=ref_image_path,
        mask_paths=ref_mask_path,
        categories=["object"],
        category_ids=np.array([1]),
    )
    model_baseline.fit(ref_sample_baseline)
    results["without_negative"] = model_baseline.predict(target_image_paths)
    logger.info(
        "Baseline: %d total masks across %d images",
        sum(p["pred_masks"].shape[0] for p in results["without_negative"]),
        len(target_image_paths),
    )

    # --- Run WITH negative prompts ---
    logger.info("Running Matcher WITH negative prompts...")
    model_negative = Matcher(device=device, num_negative_points=num_negative_points)

    # Load reference image to get dimensions for the background mask
    from instantlearn.data.utils.image import read_image

    ref_img = read_image(ref_image_path, as_tensor=False)
    h, w = ref_img.shape[:2]

    if background_mask is None and background_region is not None:
        background_mask = create_synthetic_background_mask((h, w), background_region)

    if background_mask is None:
        logger.warning("No background mask provided. Skipping negative prompt run.")
        return results

    # Load the foreground mask
    from instantlearn.data.utils.image import read_mask

    fg_mask = read_mask(ref_mask_path, as_tensor=False)

    # Create reference sample with BOTH foreground and background masks
    ref_sample_negative = Sample(
        image_path=ref_image_path,
        masks=np.stack([fg_mask, background_mask]),
        categories=["object", BACKGROUND_CATEGORY],
        category_ids=np.array([1, BACKGROUND_CATEGORY_ID]),
        is_reference=[True, True],
    )

    model_negative.fit(ref_sample_negative)

    # Verify negative embedding was extracted
    if model_negative._negative_embedding is not None:
        logger.info(
            "Extracted negative embedding: %s",
            model_negative._negative_embedding.shape,
        )
    else:
        logger.warning("No negative embedding extracted — background mask may be empty.")

    results["with_negative"] = model_negative.predict(target_image_paths)
    logger.info(
        "With negative prompts: %d total masks across %d images",
        sum(p["pred_masks"].shape[0] for p in results["with_negative"]),
        len(target_image_paths),
    )

    return results


def visualize_comparison(
    target_image_paths: list[str],
    results: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """Save side-by-side comparison images.

    Args:
        target_image_paths: Paths to target images.
        results: Dict with "without_negative" and "with_negative" predictions.
        output_dir: Directory to save visualizations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    color_map = setup_colors({1: "object"})

    for i, target_path in enumerate(target_image_paths):
        img = cv2.imread(target_path)
        if img is None:
            logger.warning("Cannot read target image: %s", target_path)
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        panels = []

        # Original
        panels.append(img_rgb.copy())

        # Without negative prompts
        if "without_negative" in results and i < len(results["without_negative"]):
            vis_baseline = render_predictions(img_rgb, results["without_negative"][i], color_map)
            n_masks = results["without_negative"][i]["pred_masks"].shape[0]
            cv2.putText(
                vis_baseline,
                f"Baseline: {n_masks} masks",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            panels.append(vis_baseline)

        # With negative prompts
        if "with_negative" in results and i < len(results["with_negative"]):
            vis_negative = render_predictions(img_rgb, results["with_negative"][i], color_map)
            n_masks = results["with_negative"][i]["pred_masks"].shape[0]
            cv2.putText(
                vis_negative,
                f"With neg: {n_masks} masks",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            panels.append(vis_negative)

        # Concatenate horizontally
        if len(panels) > 1:
            combined = np.concatenate(panels, axis=1)
        else:
            combined = panels[0]

        out_path = output_dir / f"comparison_{i:03d}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        logger.info("Saved: %s", out_path)


def visualize_reference_with_masks(
    ref_image_path: str,
    ref_mask_path: str,
    background_mask: np.ndarray | None,
    negative_points: torch.Tensor | None,
    output_dir: Path,
) -> None:
    """Visualize the reference image with foreground mask, background mask, and sampled negative points.

    Args:
        ref_image_path: Path to the reference image.
        ref_mask_path: Path to the reference foreground mask.
        background_mask: Background mask (H, W) or None.
        negative_points: Sampled negative points (M, 2) in (x, y) coords, or None.
        output_dir: Directory to save the visualization.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(ref_image_path)
    if img is None:
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    vis = img_rgb.copy()

    # Overlay foreground mask in green
    from instantlearn.data.utils.image import read_mask

    fg_mask = read_mask(ref_mask_path, as_tensor=False)
    green_overlay = np.zeros_like(vis)
    green_overlay[fg_mask > 0] = [0, 200, 0]
    vis = cv2.addWeighted(vis, 0.7, green_overlay, 0.3, 0)

    # Overlay background mask in red
    if background_mask is not None:
        red_overlay = np.zeros_like(vis)
        red_overlay[background_mask > 0] = [200, 0, 0]
        vis = cv2.addWeighted(vis, 0.85, red_overlay, 0.15, 0)

    # Draw sampled negative points as red crosses
    if negative_points is not None:
        for pt in negative_points.cpu().numpy():
            x, y = int(pt[0]), int(pt[1])
            size = max(5, img.shape[0] // 80)
            cv2.drawMarker(vis, (x, y), (255, 0, 0), cv2.MARKER_TILTED_CROSS, size, 2)

    # Labels
    cv2.putText(vis, "Green=foreground", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    cv2.putText(vis, "Red=background (negative)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0), 2)
    if negative_points is not None:
        cv2.putText(
            vis,
            f"X = sampled negative points ({negative_points.shape[0]})",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    out_path = output_dir / "reference_with_masks.jpg"
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    logger.info("Saved reference visualization: %s", out_path)


def demo_with_coco_assets(device: str = "cpu", num_negative_points: int = 5) -> None:
    """Run the demo using bundled COCO assets.

    Uses the elephant reference image + mask from the assets directory.
    A synthetic background region is placed to demonstrate false positive suppression.

    Args:
        device: Device for inference.
        num_negative_points: Points per negative mask.
    """
    ref_image = str(ASSETS_DIR / "coco" / "000000286874.jpg")
    ref_mask = str(ASSETS_DIR / "coco" / "000000286874_mask.png")
    targets = [
        str(ASSETS_DIR / "coco" / "000000390341.jpg"),
        str(ASSETS_DIR / "coco" / "000000173279.jpg"),
        str(ASSETS_DIR / "coco" / "000000267704.jpg"),
    ]

    # Verify assets exist
    for p in [ref_image, ref_mask, *targets]:
        if not Path(p).exists():
            logger.error("Asset not found: %s", p)
            logger.error("Run this script from the library/ directory.")
            return

    # Create background mask: mark a region that is NOT the elephant
    # (e.g., a grassy area or sky that might cause false positives)
    from instantlearn.data.utils.image import read_image

    ref_img = read_image(ref_image, as_tensor=False)
    h, w = ref_img.shape[:2]

    # Place background mask in the lower-right quadrant (away from the elephant)
    bg_region = (h * 3 // 4, w * 3 // 4, h, w)
    bg_mask = create_synthetic_background_mask((h, w), bg_region)

    logger.info("=" * 60)
    logger.info("Negative Prompts Demo")
    logger.info("=" * 60)
    logger.info("Reference: %s", ref_image)
    logger.info("Background region: y=[%d:%d], x=[%d:%d]", *bg_region)
    logger.info("Device: %s | Negative points: %d", device, num_negative_points)
    logger.info("=" * 60)

    # Run comparison
    results = run_comparison(
        ref_image_path=ref_image,
        ref_mask_path=ref_mask,
        target_image_paths=targets,
        background_mask=bg_mask,
        device=device,
        num_negative_points=num_negative_points,
    )

    # Visualize reference with masks and negative points
    from instantlearn.data.utils.image import read_mask

    fg_mask_np = read_mask(ref_mask, as_tensor=False)
    ref_sample = Sample(
        image_path=ref_image,
        masks=np.stack([fg_mask_np, bg_mask]),
        categories=["object", BACKGROUND_CATEGORY],
        category_ids=np.array([1, BACKGROUND_CATEGORY_ID]),
        is_reference=[True, True],
    )
    from instantlearn.components.negative_prompts import NegativeMaskToPoints

    converter = NegativeMaskToPoints(num_points_per_mask=num_negative_points)
    neg_pts, _ = converter(torch.as_tensor(bg_mask, dtype=torch.bool).unsqueeze(0))

    visualize_reference_with_masks(
        ref_image,
        ref_mask,
        bg_mask,
        neg_pts,
        OUTPUT_DIR,
    )

    # Save prediction comparisons
    visualize_comparison(targets, results, OUTPUT_DIR)

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    for i, target_path in enumerate(targets):
        name = Path(target_path).stem
        n_baseline = results["without_negative"][i]["pred_masks"].shape[0] if "without_negative" in results else 0
        n_negative = results["with_negative"][i]["pred_masks"].shape[0] if "with_negative" in results else 0
        delta = n_baseline - n_negative
        logger.info(
            "  %s: baseline=%d masks, with_neg=%d masks (delta=%+d)",
            name,
            n_baseline,
            n_negative,
            -delta,
        )
    logger.info("Output saved to: %s", OUTPUT_DIR)


def demo_multi_model(device: str = "cpu") -> None:
    """Demonstrate negative prompts across multiple models.

    Shows how the same background mask interface works for Matcher,
    PerDino, and SoftMatcher.

    Args:
        device: Device for inference.
    """
    from instantlearn.data.utils.image import read_image, read_mask
    from instantlearn.models import Matcher
    from instantlearn.models.per_dino import PerDino

    ref_image = str(ASSETS_DIR / "coco" / "000000286874.jpg")
    ref_mask = str(ASSETS_DIR / "coco" / "000000286874_mask.png")
    target = str(ASSETS_DIR / "coco" / "000000390341.jpg")

    for p in [ref_image, ref_mask, target]:
        if not Path(p).exists():
            logger.error("Asset not found: %s. Run from library/ directory.", p)
            return

    ref_img = read_image(ref_image, as_tensor=False)
    h, w = ref_img.shape[:2]
    fg_mask = read_mask(ref_mask, as_tensor=False)
    bg_mask = create_synthetic_background_mask((h, w), (h * 3 // 4, w * 3 // 4, h, w))

    # The same sample works across all models
    ref_sample = Sample(
        image_path=ref_image,
        masks=np.stack([fg_mask, bg_mask]),
        categories=["object", BACKGROUND_CATEGORY],
        category_ids=np.array([1, BACKGROUND_CATEGORY_ID]),
        is_reference=[True, True],
    )

    models = {
        "Matcher": Matcher(device=device, num_negative_points=5),
        "PerDino": PerDino(device=device, num_negative_points=5),
    }

    logger.info("=" * 60)
    logger.info("Multi-Model Negative Prompts Demo")
    logger.info("=" * 60)

    for name, model in models.items():
        model.fit(ref_sample)
        neg_emb = model._negative_embedding
        logger.info(
            "  %s: negative embedding %s",
            name,
            neg_emb.shape if neg_emb is not None else None,
        )

        predictions = model.predict(target)
        n_masks = predictions[0]["pred_masks"].shape[0]
        logger.info("  %s: predicted %d masks on target", name, n_masks)


def main() -> None:
    """Parse arguments and run the demo."""
    parser = argparse.ArgumentParser(description="Negative Prompts Demo")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference (cpu, cuda, xpu). Default: cpu",
    )
    parser.add_argument(
        "--num-negative-points",
        type=int,
        default=5,
        help="Number of negative points per background mask. Default: 5",
    )
    parser.add_argument(
        "--multi-model",
        action="store_true",
        help="Run the multi-model comparison demo.",
    )
    args = parser.parse_args()

    if args.multi_model:
        demo_multi_model(device=args.device)
    else:
        demo_with_coco_assets(
            device=args.device,
            num_negative_points=args.num_negative_points,
        )


if __name__ == "__main__":
    main()
