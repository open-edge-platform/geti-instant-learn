#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test negative prompts with real COCO potato dataset.

Tests that negative prompts (background masks) correctly suppress false
positives WITHOUT hurting true positive detections.

Usage:
    cd library
    python examples/test_negative_prompts_real.py
    python examples/test_negative_prompts_real.py --models Matcher SAM3
    python examples/test_negative_prompts_real.py --models Matcher --targets 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from instantlearn.data import Sample
from instantlearn.data.base.sample import BACKGROUND_CATEGORY, BACKGROUND_CATEGORY_ID

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATASET_DIR = Path(os.environ.get("COCO_DIR", "data/COCO"))


def load_coco_dataset(dataset_dir: Path) -> dict:
    """Load COCO annotations and return organized data."""
    ann_file = dataset_dir / "annotations" / "instances_default.json"
    with open(ann_file) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annotations: dict[int, list] = {}
    for ann in coco["annotations"]:
        annotations.setdefault(ann["image_id"], []).append(ann)

    return {"images": images, "annotations": annotations, "categories": coco["categories"]}


def coco_seg_to_mask(segmentation: list, h: int, w: int) -> np.ndarray:
    """Convert COCO polygon segmentation to binary mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def create_background_mask(
    h: int,
    w: int,
    foreground_masks: list[np.ndarray],
    strategy: str = "invert",
) -> np.ndarray:
    """Create a background mask from foreground annotations.

    Args:
        h, w: Image dimensions.
        foreground_masks: List of binary foreground masks.
        strategy: "invert" (everything not fg), "corner" (top-left 20%).

    Returns:
        Binary background mask (H, W).
    """
    combined_fg = np.zeros((h, w), dtype=np.uint8)
    for m in foreground_masks:
        combined_fg = np.maximum(combined_fg, m)

    if strategy == "invert":
        return (1 - combined_fg).astype(np.uint8)

    if strategy == "corner":
        corner = np.zeros((h, w), dtype=np.uint8)
        corner[: h // 5, : w // 5] = 1
        corner[combined_fg > 0] = 0
        return corner

    msg = f"Unknown strategy: {strategy}"
    raise ValueError(msg)


def build_model(model_name: str, device: str):
    """Instantiate a model by name."""
    if model_name == "Matcher":
        from instantlearn.models import Matcher

        return Matcher(device=device, num_negative_points=5)
    if model_name == "SAM3":
        from instantlearn.models.sam3.sam3 import SAM3

        return SAM3(device=device, num_negative_points=5)
    msg = f"Unknown model: {model_name}"
    raise ValueError(msg)


def run_model(model_name: str, ref_sample: Sample, target_paths: list[str], device: str) -> list[dict]:
    """Fit + predict with a fresh model instance."""
    model = build_model(model_name, device)
    model.fit(ref_sample)
    return model.predict(target_paths)


def run_comparison(
    model_name: str,
    ref_path: str,
    ref_masks: list[np.ndarray],
    target_paths: list[str],
    bg_mask: np.ndarray | None,
    device: str,
) -> dict:
    """Run baseline vs negative-prompts and return mask counts + scores."""
    # Baseline
    ref_baseline = Sample(
        image_path=ref_path,
        masks=np.stack(ref_masks),
        categories=["Potatoes"] * len(ref_masks),
        category_ids=np.array([1] * len(ref_masks)),
        is_reference=[True] * len(ref_masks),
    )
    preds_b = run_model(model_name, ref_baseline, target_paths, device)
    b_counts = [p["pred_masks"].shape[0] for p in preds_b]
    b_scores = [p["pred_scores"].tolist() for p in preds_b]

    # With negative
    n_counts = None
    n_scores = None
    if bg_mask is not None:
        all_masks = ref_masks + [bg_mask]
        ref_neg = Sample(
            image_path=ref_path,
            masks=np.stack(all_masks),
            categories=["Potatoes"] * len(ref_masks) + [BACKGROUND_CATEGORY],
            category_ids=np.array([1] * len(ref_masks) + [BACKGROUND_CATEGORY_ID]),
            is_reference=[True] * len(all_masks),
        )
        preds_n = run_model(model_name, ref_neg, target_paths, device)
        n_counts = [p["pred_masks"].shape[0] for p in preds_n]
        n_scores = [p["pred_scores"].tolist() for p in preds_n]

    return {
        "baseline_counts": b_counts,
        "baseline_scores": b_scores,
        "neg_counts": n_counts,
        "neg_scores": n_scores,
    }


def print_comparison(
    result: dict,
    target_names: list[str],
    strategy: str,
) -> bool:
    """Print comparison table and return True if test passes."""
    b_counts = result["baseline_counts"]
    n_counts = result["neg_counts"]
    b_scores = result["baseline_scores"]
    n_scores = result["neg_scores"]

    total_b = sum(b_counts)
    total_n = sum(n_counts) if n_counts else 0

    logger.info(
        "  %-15s  %10s  %10s  %8s  %12s  %12s",
        "Image",
        "Baseline",
        "With Neg",
        "Delta",
        "Base AvgScr",
        "Neg AvgScr",
    )
    logger.info("  " + "-" * 75)
    for i, name in enumerate(target_names):
        nb, nn = b_counts[i], n_counts[i] if n_counts else 0
        avg_b = np.mean(b_scores[i]) if b_scores[i] else 0
        avg_n = np.mean(n_scores[i]) if n_scores and n_scores[i] else 0
        logger.info("  %-15s  %10d  %10d  %+8d  %12.3f  %12.3f", name, nb, nn, nn - nb, avg_b, avg_n)

    logger.info("  " + "-" * 75)
    logger.info("  %-15s  %10d  %10d  %+8d", "TOTAL", total_b, total_n, total_n - total_b)

    if total_b == 0:
        logger.warning("  No baseline detections — cannot evaluate.")
        return True

    drop_pct = 100 * (total_b - total_n) / total_b
    avg_b_all = np.mean([s for scores in b_scores for s in scores]) if any(b_scores) else 0
    avg_n_all = np.mean([s for scores in n_scores for s in scores]) if n_scores and any(n_scores) else 0
    score_drop = avg_b_all - avg_n_all

    passed = True
    if drop_pct > 20:
        logger.error("  FAIL: Detection drop %.1f%% exceeds 20%% threshold", drop_pct)
        passed = False
    elif score_drop > 0.1:
        logger.warning("  WARN: Average score dropped by %.3f (>0.1)", score_drop)
        passed = False
    else:
        logger.info("  PASS: strategy=%s | det_change=%+.1f%% | score_change=%+.3f", strategy, -drop_pct, -score_drop)

    return passed


def main() -> None:
    """Run the full test suite."""
    parser = argparse.ArgumentParser(description="Test negative prompts with real data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="+", default=["Matcher", "SAM3"])
    parser.add_argument("--targets", type=int, default=9, help="Number of target images (max 9)")
    parser.add_argument("--ref-masks", type=int, default=2, help="Number of reference masks to use")
    args = parser.parse_args()

    coco = load_coco_dataset(DATASET_DIR)
    img_dir = DATASET_DIR / "images" / "default"
    img_ids = sorted(coco["images"].keys())

    ref_id = img_ids[0]
    target_ids = img_ids[1 : 1 + min(args.targets, len(img_ids) - 1)]

    ref_info = coco["images"][ref_id]
    ref_path = str(img_dir / ref_info["file_name"])
    h, w = ref_info["height"], ref_info["width"]

    ref_anns = coco["annotations"].get(ref_id, [])
    ref_masks = [coco_seg_to_mask(ann["segmentation"], h, w) for ann in ref_anns[: args.ref_masks]]
    target_paths = [str(img_dir / coco["images"][tid]["file_name"]) for tid in target_ids]
    target_names = [coco["images"][tid]["file_name"] for tid in target_ids]

    strategies = ["invert", "corner"]
    all_passed = True

    logger.info("=" * 80)
    logger.info("Negative Prompts — Real Data Validation")
    logger.info("=" * 80)
    logger.info("Dataset: %s", DATASET_DIR)
    logger.info("Reference: %s (%d fg masks)", ref_info["file_name"], len(ref_masks))
    logger.info("Targets: %d images", len(target_paths))
    logger.info("Models: %s", ", ".join(args.models))
    logger.info("Strategies: %s", ", ".join(strategies))

    for model_name in args.models:
        logger.info("\n" + "=" * 80)
        logger.info("MODEL: %s", model_name)
        logger.info("=" * 80)

        for strategy in strategies:
            bg_mask = create_background_mask(h, w, ref_masks, strategy=strategy)
            bg_pct = 100 * bg_mask.sum() / (h * w)
            logger.info("\n  Strategy: %s (%.1f%% of image)", strategy, bg_pct)

            result = run_comparison(model_name, ref_path, ref_masks, target_paths, bg_mask, args.device)
            passed = print_comparison(result, target_names, strategy)
            if not passed:
                all_passed = False

    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info("ALL TESTS PASSED")
    else:
        logger.error("SOME TESTS FAILED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
