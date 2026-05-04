# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unified SAM3 benchmark: latency + accuracy across backends, devices, and prompt modes.

Benchmarks SAM3 models on COCO-format datasets in both TEXT (CLASSIC) and
VISUAL_EXEMPLAR prompt modes. Measures latency (ms, FPS) and accuracy (F1@0.5,
mean IoU, avg predictions per image).

**Backends supported:**
  - PyTorch (SAM3)
  - OpenVINO (SAM3OpenVINO) — all variants: FP16, FP32, INT8_SYM, INT8_ASYM,
    INT4_SYM, INT4_ASYM

**Devices supported:**
  - CPU
  - CUDA (PyTorch only — OV models do not run on CUDA)
  - XPU / Intel GPU (PyTorch via ``xpu``, OpenVINO via ``GPU`` device)

**Output metrics:**
  - Latency: mean/std (ms), FPS
  - Accuracy: F1@0.5, mean IoU, avg predictions per image

Usage:
    # Run all OV variants on CPU (default)
    python -m instantlearn.scripts.sam3.benchmark_sam3

    # Run PyTorch on CUDA
    python -m instantlearn.scripts.sam3.benchmark_sam3 --backend pytorch --device cuda

    # Run specific OV variants
    python -m instantlearn.scripts.sam3.benchmark_sam3 --backend openvino --variants int8_sym int4_sym

    # Run on Intel GPU (XPU)
    python -m instantlearn.scripts.sam3.benchmark_sam3 --backend openvino --device xpu

    # Custom dataset directory
    python -m instantlearn.scripts.sam3.benchmark_sam3 --data-root /path/to/coco/datasets

    # Specific datasets only
    python -m instantlearn.scripts.sam3.benchmark_sam3 --datasets Potatoes Candies
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# === Defaults ===
DEFAULT_DATA_ROOT = Path("/home/devuser/workspace/data/instant_learn/prompt/geti_datasets/COCO")
DEFAULT_DATASETS: dict[str, list[str]] = {
    "Potatoes": ["Potatoes"],
    "Candies": ["Candy"],
    "Nuts": ["HazelNut", "Wallnut"],
}
CONFIDENCE_THRESHOLD = 0.5
N_WARMUP = 3

# All OpenVINO variant names (matches SAM3OVVariant enum values)
ALL_OV_VARIANTS = ["fp16", "fp32", "int8_sym", "int8_asym", "int4_sym", "int4_asym"]


# ===========================================================================
# Data structures
# ===========================================================================


@dataclass
class BenchmarkResult:
    """Result for a single (backend, variant, mode, dataset) combination."""

    backend: str
    variant: str
    device: str
    mode: str
    dataset: str
    n_images: int
    mean_latency_ms: float
    std_latency_ms: float
    fps: float
    mean_f1: float
    mean_iou: float
    avg_predictions: float
    load_time_s: float


# ===========================================================================
# Dataset loading
# ===========================================================================


def load_coco_dataset(data_root: Path, dataset_name: str) -> tuple[list[dict], dict[int, str], list[dict]]:
    """Load COCO annotations and return (images, category_id->name map, annotations)."""
    ann_path = data_root / dataset_name / "annotations" / "instances_default.json"
    with ann_path.open(encoding="utf-8") as f:
        data = json.load(f)
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    return data["images"], cat_map, data["annotations"]


def get_gt_masks_for_image(
    image_id: int, annotations: list[dict], img_h: int, img_w: int,
) -> np.ndarray:
    """Get ground truth binary masks for an image as (N, H, W) array."""
    from pycocotools import mask as mask_utils  # noqa: PLC0415

    masks = []
    for ann in annotations:
        if ann["image_id"] != image_id:
            continue
        if "segmentation" not in ann:
            continue
        seg = ann["segmentation"]
        if isinstance(seg, list):
            rle = mask_utils.frPyObjects(seg, img_h, img_w)
            rle = mask_utils.merge(rle)
        elif isinstance(seg, dict):
            rle = seg
        else:
            continue
        masks.append(mask_utils.decode(rle))
    if masks:
        return np.stack(masks)
    return np.zeros((0, img_h, img_w), dtype=np.uint8)


# ===========================================================================
# Metrics
# ===========================================================================


def compute_iou_matrix(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix between predicted and GT masks (both bool arrays)."""
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return np.zeros((len(pred_masks), len(gt_masks)))
    n_pred, n_gt = len(pred_masks), len(gt_masks)
    iou_mat = np.zeros((n_pred, n_gt))
    for i in range(n_pred):
        for j in range(n_gt):
            inter = (pred_masks[i] & gt_masks[j]).sum()
            union = (pred_masks[i] | gt_masks[j]).sum()
            iou_mat[i, j] = inter / union if union > 0 else 0.0
    return iou_mat


def compute_f1_at_iou(iou_matrix: np.ndarray, iou_threshold: float = 0.5) -> float:
    """Compute F1 score at given IoU threshold via greedy matching."""
    if iou_matrix.size == 0:
        return 0.0
    n_pred, n_gt = iou_matrix.shape
    gt_matched = np.zeros(n_gt, dtype=bool)
    tp = 0
    for i in range(n_pred):
        best_iou, best_j = 0.0, -1
        for j in range(n_gt):
            if not gt_matched[j] and iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_j = j
        if best_iou >= iou_threshold and best_j >= 0:
            tp += 1
            gt_matched[best_j] = True
    prec = tp / n_pred if n_pred > 0 else 0.0
    rec = tp / n_gt if n_gt > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def compute_mean_iou(iou_matrix: np.ndarray, iou_threshold: float = 0.3) -> float:
    """Compute mean IoU of matched predictions (greedy, threshold for matching)."""
    if iou_matrix.size == 0:
        return 0.0
    n_pred, n_gt = iou_matrix.shape
    gt_matched = np.zeros(n_gt, dtype=bool)
    matched_ious = []
    for i in range(n_pred):
        best_iou, best_j = 0.0, -1
        for j in range(n_gt):
            if not gt_matched[j] and iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_j = j
        if best_iou >= iou_threshold and best_j >= 0:
            matched_ious.append(best_iou)
            gt_matched[best_j] = True
    return float(np.mean(matched_ious)) if matched_ious else 0.0


def evaluate_predictions(
    pred_masks: torch.Tensor, gt_masks: np.ndarray, img_h: int, img_w: int,
) -> tuple[float, float, int]:
    """Evaluate predictions against GT masks.

    Returns:
        (f1, mean_iou, n_predictions)
    """
    n_preds = len(pred_masks)
    if n_preds == 0 or len(gt_masks) == 0:
        return 0.0, 0.0, n_preds

    # Resize predictions if needed
    if pred_masks.shape[-2:] != (img_h, img_w):
        pred_masks = torch.nn.functional.interpolate(
            pred_masks.unsqueeze(0).float(),
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False,
        )[0]

    pred_binary = (pred_masks > 0.5).numpy().astype(bool)
    gt_binary = gt_masks.astype(bool)

    iou_mat = compute_iou_matrix(pred_binary, gt_binary)
    f1 = compute_f1_at_iou(iou_mat, iou_threshold=0.5)
    miou = compute_mean_iou(iou_mat, iou_threshold=0.3)
    return f1, miou, n_preds


# ===========================================================================
# Model loading
# ===========================================================================


def _map_device_for_openvino(device: str) -> str:
    """Map user-friendly device names to OpenVINO device strings."""
    mapping = {
        "cpu": "CPU",
        "xpu": "GPU",
        "gpu": "GPU",
        "cuda": "GPU",  # fallback — OV doesn't do CUDA but map anyway
        "auto": "AUTO",
    }
    return mapping.get(device.lower(), device.upper())


def load_model(
    backend: str,
    variant: str,
    device: str,
    prompt_mode: str,
) -> object:
    """Load a SAM3 model for the given backend/variant/device/mode.

    Args:
        backend: "pytorch" or "openvino"
        variant: For OV: one of ALL_OV_VARIANTS. For PyTorch: ignored.
        device: Device name (cpu, cuda, xpu).
        prompt_mode: "text" or "visual_exemplar"

    Returns:
        Loaded model instance.
    """
    from instantlearn.models.sam3 import Sam3PromptMode  # noqa: PLC0415

    mode = Sam3PromptMode.CLASSIC if prompt_mode == "text" else Sam3PromptMode.VISUAL_EXEMPLAR

    if backend == "pytorch":
        from instantlearn.models import SAM3  # noqa: PLC0415

        return SAM3(
            device=device,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            prompt_mode=mode,
        )
    from instantlearn.models import SAM3OpenVINO  # noqa: PLC0415
    from instantlearn.models.sam3 import SAM3OVVariant  # noqa: PLC0415

    ov_variant = SAM3OVVariant(f"openvino-{variant}")
    ov_device = _map_device_for_openvino(device)
    return SAM3OpenVINO(
        variant=ov_variant,
        device=ov_device,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        prompt_mode=mode,
    )


# ===========================================================================
# Benchmark runners
# ===========================================================================


def benchmark_text_mode(
    model: object,
    data_root: Path,
    dataset_name: str,
    categories: list[str],
) -> tuple[list[float], list[float], list[float], list[int]]:
    """Benchmark in text-prompt (CLASSIC) mode.

    Returns:
        (latencies_ms, f1_scores, iou_scores, n_predictions_per_image)
    """
    from instantlearn.data import Sample  # noqa: PLC0415

    images, _, annotations = load_coco_dataset(data_root, dataset_name)
    img_dir = data_root / dataset_name / "images" / "default"

    # Fit with categories (for PyTorch model)
    fit_sample = Sample(
        image_path=str(img_dir / images[0]["file_name"]),
        categories=categories,
        category_ids=list(range(len(categories))),
    )
    model.fit(fit_sample)

    # Warmup
    warmup_sample = Sample(
        image_path=str(img_dir / images[0]["file_name"]),
        categories=categories,
        category_ids=list(range(len(categories))),
    )
    for _ in range(N_WARMUP):
        model.predict(warmup_sample)

    # Benchmark
    latencies, f1_scores, iou_scores, pred_counts = [], [], [], []

    for img_info in images:
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            continue

        sample = Sample(
            image_path=str(img_path),
            categories=categories,
            category_ids=list(range(len(categories))),
        )

        t_start = time.perf_counter()
        preds = model.predict(sample)
        latencies.append((time.perf_counter() - t_start) * 1000)

        pred_masks = preds[0]["pred_masks"]

        img_h = img_info.get("height", 0)
        img_w = img_info.get("width", 0)
        if img_h == 0 or img_w == 0:
            img = cv2.imread(str(img_path))
            img_h, img_w = img.shape[:2]

        gt_masks = get_gt_masks_for_image(img_info["id"], annotations, img_h, img_w)
        f1, miou, n_preds = evaluate_predictions(pred_masks, gt_masks, img_h, img_w)
        f1_scores.append(f1)
        iou_scores.append(miou)
        pred_counts.append(n_preds)

    return latencies, f1_scores, iou_scores, pred_counts


def benchmark_visual_mode(
    model: object,
    data_root: Path,
    dataset_name: str,
    categories: list[str],
) -> tuple[list[float], list[float], list[float], list[int]]:
    """Benchmark in visual-exemplar mode (fit on 1st image, predict on rest).

    Returns:
        (latencies_ms, f1_scores, iou_scores, n_predictions_per_image)
    """
    from instantlearn.data import Sample  # noqa: PLC0415

    images, cat_map, annotations = load_coco_dataset(data_root, dataset_name)
    img_dir = data_root / dataset_name / "images" / "default"

    # Use first image as reference — get one bbox per category
    ref_img_info = images[0]
    ref_path = img_dir / ref_img_info["file_name"]

    ref_bboxes, fit_cats, fit_cat_ids = [], [], []
    seen_cats: set[str] = set()
    for ann in annotations:
        if ann["image_id"] != ref_img_info["id"]:
            continue
        cat_name = cat_map[ann["category_id"]]
        if cat_name in seen_cats:
            continue
        x, y, w, h = ann["bbox"]
        ref_bboxes.append([x, y, x + w, y + h])
        fit_cats.append(cat_name)
        fit_cat_ids.append(categories.index(cat_name) if cat_name in categories else 0)
        seen_cats.add(cat_name)

    if not ref_bboxes:
        logger.warning("No annotations on reference image for %s — skipping visual mode", dataset_name)
        return [], [], [], []

    ref_sample = Sample(
        image_path=str(ref_path),
        bboxes=np.array(ref_bboxes),
        categories=fit_cats,
        category_ids=fit_cat_ids,
    )
    model.fit(ref_sample)

    # Warmup on second image
    test_images = images[1:]
    if test_images:
        warmup_sample = Sample(image_path=str(img_dir / test_images[0]["file_name"]))
        for _ in range(N_WARMUP):
            model.predict(warmup_sample)

    # Benchmark
    latencies, f1_scores, iou_scores, pred_counts = [], [], [], []

    for img_info in test_images:
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            continue

        sample = Sample(image_path=str(img_path))

        t_start = time.perf_counter()
        preds = model.predict(sample)
        latencies.append((time.perf_counter() - t_start) * 1000)

        pred_masks = preds[0]["pred_masks"]

        img_h = img_info.get("height", 0)
        img_w = img_info.get("width", 0)
        if img_h == 0 or img_w == 0:
            img = cv2.imread(str(img_path))
            img_h, img_w = img.shape[:2]

        gt_masks = get_gt_masks_for_image(img_info["id"], annotations, img_h, img_w)
        f1, miou, n_preds = evaluate_predictions(pred_masks, gt_masks, img_h, img_w)
        f1_scores.append(f1)
        iou_scores.append(miou)
        pred_counts.append(n_preds)

    return latencies, f1_scores, iou_scores, pred_counts


# ===========================================================================
# Main orchestration
# ===========================================================================


def run_benchmark(
    backend: str,
    variants: list[str],
    device: str,
    data_root: Path,
    datasets: dict[str, list[str]],
    modes: list[str],
) -> list[BenchmarkResult]:
    """Run the full benchmark across variants, datasets, and modes."""
    results: list[BenchmarkResult] = []

    for variant in variants:
        variant_label = f"{backend}/{variant}" if backend == "openvino" else f"{backend}"

        for mode in modes:
            for dataset_name, categories in datasets.items():
                logger.info(
                    "Benchmarking: %s | %s | %s | device=%s",
                    variant_label, mode, dataset_name, device,
                )

                # Check device compatibility
                if backend == "openvino" and device.lower() == "cuda":
                    logger.warning(
                        "OpenVINO does not support CUDA. Skipping %s on CUDA.", variant,
                    )
                    continue

                try:
                    t0 = time.perf_counter()
                    model = load_model(backend, variant, device, mode)
                    load_time = time.perf_counter() - t0
                except Exception:
                    logger.exception("Failed to load %s/%s on %s", backend, variant, device)
                    continue

                try:
                    if mode == "text":
                        latencies, f1_scores, iou_scores, pred_counts = benchmark_text_mode(
                            model, data_root, dataset_name, categories,
                        )
                    else:
                        latencies, f1_scores, iou_scores, pred_counts = benchmark_visual_mode(
                            model, data_root, dataset_name, categories,
                        )
                except Exception:
                    logger.exception(
                        "Error benchmarking %s/%s on %s/%s", backend, variant, mode, dataset_name,
                    )
                    continue
                finally:
                    del model
                    gc.collect()
                    if device.lower() == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if not latencies:
                    continue

                mean_lat = float(np.mean(latencies))
                results.append(BenchmarkResult(
                    backend=backend,
                    variant=variant,
                    device=device,
                    mode=mode,
                    dataset=dataset_name,
                    n_images=len(latencies),
                    mean_latency_ms=mean_lat,
                    std_latency_ms=float(np.std(latencies)),
                    fps=1000.0 / mean_lat if mean_lat > 0 else 0.0,
                    mean_f1=float(np.mean(f1_scores)) if f1_scores else 0.0,
                    mean_iou=float(np.mean(iou_scores)) if iou_scores else 0.0,
                    avg_predictions=float(np.mean(pred_counts)) if pred_counts else 0.0,
                    load_time_s=load_time,
                ))

                logger.info(
                    "  → lat=%.0fms  fps=%.1f  F1=%.3f  mIoU=%.3f  avgPreds=%.1f",
                    results[-1].mean_latency_ms,
                    results[-1].fps,
                    results[-1].mean_f1,
                    results[-1].mean_iou,
                    results[-1].avg_predictions,
                )

    return results


def print_results_table(results: list[BenchmarkResult]) -> None:
    """Print results as a formatted table."""
    if not results:
        print("No results to display.")
        return

    header = (
        f"{'Backend':<10} {'Variant':<10} {'Device':<6} {'Mode':<16} {'Dataset':<10} "
        f"{'Imgs':<5} {'Lat(ms)':<10} {'FPS':<7} {'F1@0.5':<7} {'mIoU':<7} {'AvgPred':<8} {'Load(s)':<7}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("SAM3 BENCHMARK RESULTS")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in results:
        print(
            f"{r.backend:<10} {r.variant:<10} {r.device:<6} {r.mode:<16} {r.dataset:<10} "
            f"{r.n_images:<5} {r.mean_latency_ms:>5.0f}±{r.std_latency_ms:<3.0f} "
            f"{r.fps:<7.1f} {r.mean_f1:<7.3f} {r.mean_iou:<7.3f} {r.avg_predictions:<8.1f} {r.load_time_s:<7.1f}",
        )

    # Aggregated per variant+mode
    print("\n" + sep)
    print("AGGREGATED (mean across datasets)")
    print(sep)
    agg_hdr = (
        f"{'Backend':<10} {'Variant':<10} {'Device':<6} {'Mode':<16} "
        f"{'Lat(ms)':<10} {'FPS':<7} {'F1@0.5':<7} {'mIoU':<7} {'AvgPred':<8}"
    )
    print(agg_hdr)
    print(sep)

    # Group by (backend, variant, device, mode)
    groups: dict[tuple, list[BenchmarkResult]] = {}
    for r in results:
        key = (r.backend, r.variant, r.device, r.mode)
        groups.setdefault(key, []).append(r)

    for (backend, variant, device, mode), group in sorted(groups.items()):
        avg_lat = np.mean([r.mean_latency_ms for r in group])
        avg_fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0
        avg_f1 = np.mean([r.mean_f1 for r in group])
        avg_iou = np.mean([r.mean_iou for r in group])
        avg_pred = np.mean([r.avg_predictions for r in group])
        print(
            f"{backend:<10} {variant:<10} {device:<6} {mode:<16} "
            f"{avg_lat:>5.0f}     {avg_fps:<7.1f} {avg_f1:<7.3f} {avg_iou:<7.3f} {avg_pred:<8.1f}",
        )


def save_results_json(results: list[BenchmarkResult], output_path: Path) -> None:
    """Save results to a JSON file."""
    data = [
        {
            "backend": r.backend,
            "variant": r.variant,
            "device": r.device,
            "mode": r.mode,
            "dataset": r.dataset,
            "n_images": r.n_images,
            "mean_latency_ms": r.mean_latency_ms,
            "std_latency_ms": r.std_latency_ms,
            "fps": r.fps,
            "mean_f1": r.mean_f1,
            "mean_iou": r.mean_iou,
            "avg_predictions": r.avg_predictions,
            "load_time_s": r.load_time_s,
        }
        for r in results
    ]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Results saved to %s", output_path)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Unified SAM3 benchmark: latency + accuracy across backends, devices, and modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # All OV variants on CPU
  python -m instantlearn.scripts.sam3.benchmark_sam3

  # PyTorch on CUDA
  python -m instantlearn.scripts.sam3.benchmark_sam3 --backend pytorch --device cuda

  # Specific OV variants on Intel GPU
  python -m instantlearn.scripts.sam3.benchmark_sam3 --variants int8_sym int4_sym --device xpu

  # Only text mode
  python -m instantlearn.scripts.sam3.benchmark_sam3 --modes text

  # Only visual exemplar mode
  python -m instantlearn.scripts.sam3.benchmark_sam3 --modes visual_exemplar
""",
    )
    parser.add_argument(
        "--backend",
        choices=["openvino", "pytorch"],
        default="openvino",
        help="Model backend. Default: openvino",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help=(
            "OV variants to benchmark (e.g. int8_sym int4_asym). "
            "Default: all available variants. Ignored for pytorch backend."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device: cpu, cuda, xpu. Default: cpu",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["text", "visual_exemplar"],
        default=["text", "visual_exemplar"],
        help="Prompt modes to benchmark. Default: both text and visual_exemplar",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Root directory containing COCO datasets. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to benchmark (subdirectory names under data-root). Default: all configured.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=N_WARMUP,
        help=f"Number of warmup iterations. Default: {N_WARMUP}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save JSON results. Default: benchmark_results_<timestamp>.json",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the unified SAM3 benchmark."""
    args = parse_args()

    global N_WARMUP  # noqa: PLW0603
    N_WARMUP = args.warmup

    # Resolve datasets
    if args.datasets:
        datasets = {}
        for ds_name in args.datasets:
            if ds_name in DEFAULT_DATASETS:
                datasets[ds_name] = DEFAULT_DATASETS[ds_name]
            else:
                # Auto-discover categories from annotation file
                ann_path = args.data_root / ds_name / "annotations" / "instances_default.json"
                if ann_path.exists():
                    with ann_path.open(encoding="utf-8") as f:
                        data = json.load(f)
                    datasets[ds_name] = [c["name"] for c in data["categories"]]
                else:
                    logger.warning("Dataset %s not found at %s", ds_name, ann_path)
    else:
        datasets = DEFAULT_DATASETS

    # Resolve variants
    if args.backend == "pytorch":
        variants = ["pytorch"]
    elif args.variants:
        variants = args.variants
    else:
        variants = ALL_OV_VARIANTS

    # Log configuration
    logger.info("=" * 70)
    logger.info("SAM3 Unified Benchmark")
    logger.info("=" * 70)
    logger.info("  Backend:  %s", args.backend)
    logger.info("  Variants: %s", variants)
    logger.info("  Device:   %s", args.device)
    logger.info("  Modes:    %s", args.modes)
    logger.info("  Datasets: %s", list(datasets.keys()))
    logger.info("  Warmup:   %d iterations", N_WARMUP)
    logger.info("=" * 70)

    # Run
    results = run_benchmark(
        backend=args.backend,
        variants=variants,
        device=args.device,
        data_root=args.data_root,
        datasets=datasets,
        modes=args.modes,
    )

    # Display
    print_results_table(results)

    # Save
    if args.output:
        output_path = args.output
    else:
        from datetime import datetime, timezone  # noqa: PLC0415

        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"benchmark_results_{ts}.json")

    save_results_json(results, output_path)


if __name__ == "__main__":
    main()
