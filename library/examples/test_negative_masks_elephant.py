#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test negative masks with PerSeg elephant dataset on Matcher, SoftMatcher, SAM3.

Uses blob-shaped negative regions (not perfect squares) to simulate
realistic user-drawn background annotations.

Usage:
    cd library
    python examples/test_negative_masks_elephant.py
    python examples/test_negative_masks_elephant.py --models Matcher SoftMatcher SAM3
    python examples/test_negative_masks_elephant.py --models Matcher --targets 2
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import cv2
import numpy as np
from PIL import Image

from instantlearn.data import Sample
from instantlearn.data.base.sample import BACKGROUND_CATEGORY, BACKGROUND_CATEGORY_ID

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATASET_DIR = os.environ.get("PERSEG_DIR", "data/PerSeg")
CATEGORY = "elephant"


def load_perseg_elephant() -> dict:
    """Load PerSeg elephant images and annotations."""
    img_dir = f"{DATASET_DIR}/Images/{CATEGORY}"
    ann_dir = f"{DATASET_DIR}/Annotations/{CATEGORY}"

    images = []
    for i in range(5):
        img_path = f"{img_dir}/{i:02d}.jpg"
        ann_path = f"{ann_dir}/{i:02d}.png"
        img = np.array(Image.open(img_path).convert("RGB"))
        ann = np.array(Image.open(ann_path))
        mask = (ann > 0).astype(np.uint8)
        images.append({
            "path": img_path,
            "image": img,
            "mask": mask,
            "height": img.shape[0],
            "width": img.shape[1],
            "name": f"{i:02d}.jpg",
        })
    return images


def generate_random_blobs(
    h: int,
    w: int,
    num_blobs: int = 5,
    min_radius: int = 30,
    max_radius: int = 80,
    seed: int = 42,
) -> np.ndarray:
    """Generate a mask with random blob shapes (ellipses + gaussian blur).

    Creates organic-looking blobs at random positions, avoiding
    the center of the image (where the subject likely is).
    """
    rng = np.random.RandomState(seed)
    mask = np.zeros((h, w), dtype=np.uint8)

    for _ in range(num_blobs):
        # Place blobs in border regions (outer 30% of each edge)
        if rng.rand() < 0.5:
            cx = rng.randint(0, int(w * 0.25)) if rng.rand() < 0.5 else rng.randint(int(w * 0.75), w)
            cy = rng.randint(0, h)
        else:
            cx = rng.randint(0, w)
            cy = rng.randint(0, int(h * 0.25)) if rng.rand() < 0.5 else rng.randint(int(h * 0.75), h)

        # Random ellipse parameters
        ax1 = rng.randint(min_radius, max_radius)
        ax2 = rng.randint(min_radius, max_radius)
        angle = rng.randint(0, 360)
        cv2.ellipse(mask, (cx, cy), (ax1, ax2), angle, 0, 360, 1, -1)

    # Smooth to make blob-like
    ksize = max(15, min_radius // 2) | 1  # ensure odd
    mask = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), 0)
    mask = (mask > 0.3).astype(np.uint8)
    return mask


def create_blob_bg_mask(
    h: int,
    w: int,
    fg_mask: np.ndarray,
    strategy: str = "border_blobs",
    seed: int = 42,
) -> np.ndarray:
    """Create blob-shaped background masks.

    Strategies:
        border_blobs: Random blobs in the border/corner regions, excluding foreground.
        inverted_blobs: Invert fg mask, then erode with blobs to create organic BG.
        scattered_blobs: Large blobs scattered across the whole image, excluding foreground.
    """
    rng = np.random.RandomState(seed)

    if strategy == "border_blobs":
        # Blobs placed near edges
        mask = generate_random_blobs(h, w, num_blobs=6, min_radius=40, max_radius=120, seed=seed)
        mask[fg_mask > 0] = 0
        return mask

    if strategy == "inverted_blobs":
        # Inverted fg with blobby erosion to simulate imprecise user BG selection
        bg = (1 - fg_mask).astype(np.uint8)
        # Create random erosion kernel (blob-shaped)
        blob_holes = generate_random_blobs(h, w, num_blobs=8, min_radius=60, max_radius=200, seed=seed + 1)
        # Remove some BG areas to make it blobby/imprecise
        bg[blob_holes > 0] = 0
        bg[fg_mask > 0] = 0
        return bg

    if strategy == "scattered_blobs":
        # Larger blobs scattered widely
        mask = generate_random_blobs(h, w, num_blobs=10, min_radius=50, max_radius=150, seed=seed)
        mask[fg_mask > 0] = 0
        return mask

    msg = f"Unknown strategy: {strategy}"
    raise ValueError(msg)


def build_model(model_name: str, device: str):
    """Instantiate a model by name."""
    if model_name == "Matcher":
        from instantlearn.models import Matcher

        return Matcher(device=device, num_negative_points=5)
    if model_name == "SoftMatcher":
        from instantlearn.models.soft_matcher.soft_matcher import SoftMatcher

        return SoftMatcher(device=device, num_negative_points=5)
    if model_name == "SAM3":
        from instantlearn.models.sam3.sam3 import SAM3

        return SAM3(device=device, num_negative_points=5)
    msg = f"Unknown model: {model_name}"
    raise ValueError(msg)


def run_model(model, ref_sample: Sample, target_paths: list[str]) -> list[dict]:
    """Fit + predict."""
    model.fit(ref_sample)
    return model.predict(target_paths)


def run_comparison(
    model_name: str,
    ref_path: str,
    ref_mask: np.ndarray,
    target_paths: list[str],
    bg_mask: np.ndarray | None,
    device: str,
) -> dict:
    """Run baseline vs negative-mask comparison."""
    # Baseline
    ref_baseline = Sample(
        image_path=ref_path,
        masks=ref_mask[np.newaxis],
        categories=[CATEGORY],
        category_ids=np.array([1]),
        is_reference=[True],
    )
    model_b = build_model(model_name, device)
    t0 = time.time()
    preds_b = run_model(model_b, ref_baseline, target_paths)
    time_b = time.time() - t0
    b_counts = [p["pred_masks"].shape[0] for p in preds_b]
    b_scores = [
        list(p["pred_scores"].tolist() if hasattr(p["pred_scores"], "tolist") else p["pred_scores"]) for p in preds_b
    ]

    # With negative mask
    n_counts = None
    n_scores = None
    time_n = 0
    if bg_mask is not None:
        all_masks = np.stack([ref_mask, bg_mask])
        ref_neg = Sample(
            image_path=ref_path,
            masks=all_masks,
            categories=[CATEGORY, BACKGROUND_CATEGORY],
            category_ids=np.array([1, BACKGROUND_CATEGORY_ID]),
            is_reference=[True, True],
        )
        model_n = build_model(model_name, device)
        t0 = time.time()
        preds_n = run_model(model_n, ref_neg, target_paths)
        time_n = time.time() - t0
        n_counts = [p["pred_masks"].shape[0] for p in preds_n]
        n_scores = [
            list(p["pred_scores"].tolist() if hasattr(p["pred_scores"], "tolist") else p["pred_scores"])
            for p in preds_n
        ]

    return {
        "baseline_counts": b_counts,
        "baseline_scores": b_scores,
        "neg_counts": n_counts,
        "neg_scores": n_scores,
        "time_baseline": time_b,
        "time_neg": time_n,
    }


def print_comparison(
    result: dict,
    target_names: list[str],
    strategy: str,
) -> bool:
    """Print per-image comparison and return True if test passes."""
    b_counts = result["baseline_counts"]
    n_counts = result["neg_counts"]
    b_scores = result["baseline_scores"]
    n_scores = result["neg_scores"]

    total_b = sum(b_counts)
    total_n = sum(n_counts) if n_counts else 0

    logger.info(
        "  %-15s  %10s  %10s  %8s  %10s  %10s",
        "Image",
        "Baseline",
        "With Neg",
        "Delta",
        "BaseAvg",
        "NegAvg",
    )
    logger.info("  " + "-" * 70)
    for i, name in enumerate(target_names):
        nb = b_counts[i]
        nn = n_counts[i] if n_counts else 0
        avg_b = float(np.mean(b_scores[i])) if b_scores[i] else 0
        avg_n = float(np.mean(n_scores[i])) if n_scores and n_scores[i] else 0
        logger.info(
            "  %-15s  %10d  %10d  %+8d  %10.3f  %10.3f",
            name,
            nb,
            nn,
            nn - nb,
            avg_b,
            avg_n,
        )

    logger.info("  " + "-" * 70)
    avg_b_all = float(np.mean([s for scores in b_scores for s in scores])) if any(b_scores) else 0
    avg_n_all = float(np.mean([s for scores in n_scores for s in scores])) if n_scores and any(n_scores) else 0
    logger.info(
        "  %-15s  %10d  %10d  %+8d  %10.3f  %10.3f",
        "TOTAL",
        total_b,
        total_n,
        total_n - total_b,
        avg_b_all,
        avg_n_all,
    )
    logger.info(
        "  Time: baseline=%.1fs, with_neg=%.1fs",
        result["time_baseline"],
        result["time_neg"],
    )

    if total_b == 0:
        logger.warning("  No baseline detections — skipping.")
        return True

    drop_pct = 100 * (total_b - total_n) / total_b
    score_drop = avg_b_all - avg_n_all

    passed = True
    if drop_pct > 20:
        logger.error("  FAIL: Detection drop %.1f%% exceeds 20%% threshold", drop_pct)
        passed = False
    if score_drop > 0.1:
        logger.error("  FAIL: Score drop %.3f exceeds 0.1 threshold", score_drop)
        passed = False

    if passed:
        logger.info(
            "  PASS: strategy=%s | det_change=%+.1f%% | score_change=%+.3f",
            strategy,
            -drop_pct,
            -score_drop,
        )
    return passed


def main() -> None:
    """Run the full test."""
    parser = argparse.ArgumentParser(description="Test negative masks with elephant dataset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Matcher", "SoftMatcher", "SAM3"],
        help="Models to test (Matcher, SoftMatcher, SAM3)",
    )
    parser.add_argument("--targets", type=int, default=4, help="Number of target images (max 4)")
    args = parser.parse_args()

    images = load_perseg_elephant()
    ref = images[0]
    targets = images[1 : 1 + min(args.targets, len(images) - 1)]

    ref_path = ref["path"]
    ref_mask = ref["mask"]
    h, w = ref["height"], ref["width"]
    target_paths = [t["path"] for t in targets]
    target_names = [t["name"] for t in targets]

    strategies = ["border_blobs", "inverted_blobs", "scattered_blobs"]

    logger.info("=" * 80)
    logger.info("Negative Masks Test — PerSeg Elephant Dataset (Blob Masks)")
    logger.info("=" * 80)
    logger.info("Reference: %s (%dx%d, fg=%.1f%%)", ref["name"], w, h, 100 * ref_mask.sum() / (h * w))
    logger.info("Targets: %s", ", ".join(target_names))
    logger.info("Models: %s", ", ".join(args.models))
    logger.info("Strategies: %s", ", ".join(strategies))

    # Show blob mask stats
    logger.info("\nBlob mask coverage:")
    for strategy in strategies:
        bg = create_blob_bg_mask(h, w, ref_mask, strategy=strategy)
        pct = 100 * bg.sum() / (h * w)
        logger.info("  %-20s: %6.1f%% of image, %d pixels", strategy, pct, bg.sum())

    all_passed = True
    for model_name in args.models:
        logger.info("\n" + "=" * 80)
        logger.info("MODEL: %s", model_name)
        logger.info("=" * 80)

        for strategy in strategies:
            bg_mask = create_blob_bg_mask(h, w, ref_mask, strategy=strategy)
            bg_pct = 100 * bg_mask.sum() / (h * w)
            logger.info(
                "\n  Strategy: %s (%.1f%% of image as background)",
                strategy,
                bg_pct,
            )

            result = run_comparison(
                model_name,
                ref_path,
                ref_mask,
                target_paths,
                bg_mask,
                args.device,
            )
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
