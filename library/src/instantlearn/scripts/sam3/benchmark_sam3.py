# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unified SAM3 benchmark with two subcommands: accuracy and profile.

**Subcommand: accuracy**
Benchmarks SAM3 models on COCO-format datasets in both TEXT (CLASSIC) and
VISUAL_EXEMPLAR prompt modes. Measures latency (ms, FPS) and accuracy (F1@0.5,
mean IoU, avg predictions per image).

  Backends: PyTorch (SAM3), OpenVINO (SAM3OpenVINO)
  Devices: CPU, CUDA, XPU / Intel GPU
  Output: JSON results + console table

**Subcommand: profile**
Component-level and end-to-end latency profiling for OpenVINO SAM3 inference
across model variants, prompt types, and compile configurations.

  Variants: FP16, INT8-SYM, INT8-ASYM, INT4-SYM, INT4-ASYM
  Prompt types: text, box, point-as-box
  Scenarios: cold start, cached vision, live stream
  Output: Rich tables, Excel, PNG charts

Usage:
    # Accuracy: all OV variants on CPU
    python -m instantlearn.scripts.sam3.benchmark_sam3 accuracy

    # Accuracy: PyTorch on CUDA
    python -m instantlearn.scripts.sam3.benchmark_sam3 accuracy --backend pytorch --device cuda

    # Profile: auto-download from HuggingFace
    python -m instantlearn.scripts.sam3.benchmark_sam3 profile

    # Profile: on Intel GPU
    python -m instantlearn.scripts.sam3.benchmark_sam3 profile --device GPU

    # Profile: specific variants with more iterations
    python -m instantlearn.scripts.sam3.benchmark_sam3 profile --variants openvino-int8_sym --iterations 20
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
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



_PROFILE_HF_REPO_ID = "rajeshgangireddy/SAM3_OpenVINO"

_VISION_ENCODER = "vision-encoder"
_TEXT_ENCODER = "text-encoder"
_GEOMETRY_ENCODER = "geometry-encoder"
_GEOMETRY_ENCODER_EXEMPLAR = "geometry-encoder-exemplar"
_PROMPT_DECODER = "prompt-decoder"

_PROFILE_DEFAULT_VARIANTS: dict[str, str] = {
    "openvino-fp16": "OV-FP16",
    "openvino-int8_sym": "OV-INT8-SYM",
    "openvino-int8_asym": "OV-INT8-ASYM",
    "openvino-int4_sym": "OV-INT4-SYM",
    "openvino-int4_asym": "OV-INT4-ASYM",
}

_OV_CONFIGS_CPU: dict[str, dict] = {
    "default": {},
    "latency-hint": {"PERFORMANCE_HINT": "LATENCY"},
    "throughput-hint": {"PERFORMANCE_HINT": "THROUGHPUT"},
}

_OV_CONFIGS_GPU: dict[str, dict] = {
    "default": {"INFERENCE_PRECISION_HINT": "f16"},
    "latency-hint": {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f16"},
}


@dataclass
class TimingResult:
    """Timing for a single inference run broken down by component."""

    preprocess_ms: float = 0.0
    vision_encoder_ms: float = 0.0
    text_encoder_ms: float = 0.0
    geometry_encoder_ms: float = 0.0
    decoder_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        """Total end-to-end time in ms."""
        return (
            self.preprocess_ms + self.vision_encoder_ms + self.text_encoder_ms
            + self.geometry_encoder_ms + self.decoder_ms + self.postprocess_ms
        )

    @property
    def without_vision_ms(self) -> float:
        """Time without vision encoder (cached features scenario)."""
        return self.text_encoder_ms + self.geometry_encoder_ms + self.decoder_ms + self.postprocess_ms


@dataclass
class ProfileResult:
    """Aggregated profile results for one configuration."""

    variant: str
    config_name: str
    prompt_type: str
    timings: list[TimingResult] = field(default_factory=list)

    def _values(self, attr: str) -> list[float]:
        return [getattr(t, attr) for t in self.timings]

    def stats(self, attr: str) -> dict[str, float]:
        """Return mean, median, std, min, max for a timing attribute."""
        vals = self._values(attr)
        if not vals:
            return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}
        return {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0,
            "min": min(vals),
            "max": max(vals),
        }


def _is_gpu_available() -> bool:
    """Check whether an OpenVINO GPU device is available."""
    import openvino as ov  # noqa: PLC0415

    try:
        core = ov.Core()
        return "GPU" in core.available_devices
    except Exception:  # noqa: BLE001
        return False


def _get_ov_configs(device: str, cache_dir: Path | None = None) -> dict[str, dict]:
    """Return device-appropriate compile configs."""
    base = _OV_CONFIGS_GPU if device.upper().startswith("GPU") else _OV_CONFIGS_CPU
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        base = {name: {**cfg, "CACHE_DIR": str(cache_dir)} for name, cfg in base.items()}
    return base


def _download_variant(variant: str, repo_id: str = _PROFILE_HF_REPO_ID) -> Path:
    """Download a model variant from HuggingFace Hub."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    logger.info("Downloading %s from %s...", variant, repo_id)
    cache_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{variant}/*", "tokenizer*", "special_tokens_map*"],
    )
    return Path(cache_dir) / variant


def _resolve_variant_path(base_dir: Path | None, variant: str) -> Path | None:
    """Resolve a variant to a local directory, downloading from HF if needed."""
    if base_dir is not None:
        path = base_dir / variant
        if not path.is_dir():
            logger.warning("Skipping %s: not found in %s", variant, base_dir)
            return None
        return path
    try:
        return _download_variant(variant)
    except Exception:
        logger.exception("Failed to download %s from HuggingFace", variant)
        return None


class ProfileModel:
    """Lightweight model wrapper for benchmarking individual OpenVINO components."""

    def __init__(self, model_dir: Path, device: str = "CPU", config: dict | None = None) -> None:
        """Initialize profile model with compile configuration."""
        import openvino as ov  # noqa: PLC0415
        from transformers import CLIPTokenizerFast  # noqa: PLC0415

        from instantlearn.models.sam3.processing import (  # noqa: PLC0415
            Sam3Postprocessor,
            Sam3Preprocessor,
            Sam3PromptPreprocessor,
        )

        core = ov.Core()
        compile_config = config or {}

        names = [_VISION_ENCODER, _TEXT_ENCODER, _GEOMETRY_ENCODER, _GEOMETRY_ENCODER_EXEMPLAR, _PROMPT_DECODER]
        compiled = {}
        for name in names:
            path = self._find(model_dir, name)
            logger.info("Compiling %s on %s...", name, device)
            t0 = time.perf_counter()
            compiled[name] = core.compile_model(path, device, compile_config)
            logger.info("  %s compiled in %.1fs", name, time.perf_counter() - t0)

        self.vision_model = compiled[_VISION_ENCODER]
        self.text_model = compiled[_TEXT_ENCODER]
        self.geometry_model = compiled[_GEOMETRY_ENCODER]
        self.geometry_exemplar_model = compiled[_GEOMETRY_ENCODER_EXEMPLAR]
        self.decoder_model = compiled[_PROMPT_DECODER]

        self.vision_request = self.vision_model.create_infer_request()
        self.text_request = self.text_model.create_infer_request()
        self.geometry_request = self.geometry_model.create_infer_request()
        self.geometry_exemplar_request = self.geometry_exemplar_model.create_infer_request()
        self.decoder_request = self.decoder_model.create_infer_request()

        resolution = 1008
        self.preprocessor = Sam3Preprocessor(target_size=resolution)
        self.prompt_preprocessor = Sam3PromptPreprocessor(target_size=resolution)
        self.postprocessor = Sam3Postprocessor(target_size=resolution, threshold=0.5, mask_threshold=0.5)

        if (model_dir / "tokenizer.json").exists():
            self.tokenizer = CLIPTokenizerFast.from_pretrained(str(model_dir))
        else:
            self.tokenizer = CLIPTokenizerFast.from_pretrained(_PROFILE_HF_REPO_ID)

    @staticmethod
    def _find(model_dir: Path, name: str) -> Path:
        """Find model file (.xml preferred, then .onnx)."""
        for ext in (".xml", ".onnx", "-fp16.onnx"):
            candidate = model_dir / f"{name}{ext}"
            if candidate.exists():
                return candidate
        variants = sorted(model_dir.glob(f"{name}*.onnx"))
        if variants:
            return variants[0]
        msg = f"Model '{name}' not found in {model_dir}"
        raise FileNotFoundError(msg)

    def run_preprocess(self, image: torch.Tensor) -> tuple[np.ndarray, list[tuple[int, int]], float]:
        """Preprocess image; returns (pixel_values_np, original_sizes, elapsed_ms)."""
        t0 = time.perf_counter()
        image_4d = image.unsqueeze(0) if image.ndim == 3 else image
        with torch.no_grad():
            pixel_values, original_sizes = self.preprocessor(image_4d)
        return pixel_values.numpy(), original_sizes, (time.perf_counter() - t0) * 1000

    def run_vision_encoder(self, pixel_values: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
        """Run vision encoder; returns (features_dict, elapsed_ms)."""
        t0 = time.perf_counter()
        self.vision_request.infer([pixel_values])
        result = {k: self.vision_request.get_tensor(k).data for k in ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"]}
        return result, (time.perf_counter() - t0) * 1000

    def run_text_encoder(self, text: str) -> tuple[np.ndarray, np.ndarray, float]:
        """Run text encoder; returns (text_features, text_mask, elapsed_ms)."""
        tokens = self.tokenizer([text], return_tensors="np", padding=True)
        input_ids = _profile_pad_or_truncate(tokens.input_ids.astype(np.int64), 32)
        attention_mask = _profile_pad_or_truncate(tokens.attention_mask.astype(np.int64), 32)
        t0 = time.perf_counter()
        self.text_request.infer([input_ids, attention_mask])
        return self.text_request.get_tensor("text_features").data, self.text_request.get_tensor("text_mask").data, (time.perf_counter() - t0) * 1000

    def run_geometry_encoder(
        self, vision_features: dict[str, np.ndarray],
        input_boxes: np.ndarray, input_boxes_labels: np.ndarray,
        input_points: np.ndarray, input_points_labels: np.ndarray,
        *, exemplar: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Run geometry encoder; returns (geometry_features, geometry_mask, elapsed_ms)."""
        request = self.geometry_exemplar_request if exemplar else self.geometry_request
        t0 = time.perf_counter()
        request.infer([vision_features["fpn_feat_2"], vision_features["fpn_pos_2"],
                       input_boxes, input_boxes_labels, input_points, input_points_labels])
        return request.get_tensor("geometry_features").data, request.get_tensor("geometry_mask").data, (time.perf_counter() - t0) * 1000

    def run_prompt_decoder(
        self, vision_features: dict[str, np.ndarray],
        prompt_features: np.ndarray, prompt_mask: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float]:
        """Run prompt decoder; returns (outputs_dict, elapsed_ms)."""
        t0 = time.perf_counter()
        self.decoder_request.infer([
            vision_features["fpn_feat_0"], vision_features["fpn_feat_1"],
            vision_features["fpn_feat_2"], vision_features["fpn_pos_2"],
            prompt_features, prompt_mask,
        ])
        result = {k: self.decoder_request.get_tensor(k).data for k in ["pred_masks", "pred_boxes", "pred_logits", "presence_logits"]}
        return result, (time.perf_counter() - t0) * 1000

    def run_postprocess(self, decoder_outputs: dict[str, np.ndarray], img_size: tuple[int, int]) -> tuple[dict, float]:
        """Run postprocessing; returns (results_dict, elapsed_ms)."""
        t0 = time.perf_counter()
        outputs_torch = {k: torch.from_numpy(np.array(v)) for k, v in decoder_outputs.items()}
        with torch.no_grad():
            result = self.postprocessor(outputs_torch, target_sizes=[img_size])
        return result[0], (time.perf_counter() - t0) * 1000


def _profile_pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or truncate array to target sequence length."""
    cur = arr.shape[1]
    if cur == target_len:
        return arr
    if cur > target_len:
        return arr[:, :target_len]
    return np.concatenate([arr, np.zeros((arr.shape[0], target_len - cur), dtype=arr.dtype)], axis=1)


def _profile_make_dummy_image(h: int = 720, w: int = 1280) -> torch.Tensor:
    """Create a realistic-sized dummy image tensor [3, H, W]."""
    rng = np.random.default_rng(42)
    return torch.from_numpy(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).permute(2, 0, 1).float() / 255.0


def _profile_make_real_image(image_path: Path) -> torch.Tensor | None:
    """Load a real image, or return None if not found."""
    if not image_path.exists():
        return None
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    return torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0


def _sentinel_box() -> tuple[np.ndarray, np.ndarray]:
    """Return sentinel box inputs (no box prompt)."""
    return np.zeros((1, 1, 4), dtype=np.float32), np.full((1, 1), -10, dtype=np.int64)


def _sentinel_points() -> tuple[np.ndarray, np.ndarray]:
    """Return sentinel point inputs (no point prompt)."""
    return np.zeros((1, 1, 2), dtype=np.float32), np.full((1, 1), -10, dtype=np.int64)


def _real_box(preprocessor: object, bbox: np.ndarray, original_sizes: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert a xyxy bbox to the model's normalized cxcywh format."""
    with torch.no_grad():
        box_tensor, _ = preprocessor(original_sizes, input_boxes=bbox)
    boxes = box_tensor.numpy().astype(np.float32)
    return boxes, np.ones((1, boxes.shape[1]), dtype=np.int64)


def profile_single_inference(
    model: ProfileModel, image: torch.Tensor, text: str, bbox: np.ndarray | None = None,
) -> TimingResult:
    """Run one complete inference pass and time each component."""
    img_size = image.shape[-2:]
    pixel_np, original_sizes, t_pre = model.run_preprocess(image)
    vision_features, t_vis = model.run_vision_encoder(pixel_np)
    text_features, text_mask, t_txt = model.run_text_encoder(text)

    t_geo = 0.0
    if bbox is not None:
        input_boxes, input_boxes_labels = _real_box(model.prompt_preprocessor, bbox, original_sizes)
        input_points, input_points_labels = _sentinel_points()
        geo_features, geo_mask, t_geo = model.run_geometry_encoder(
            vision_features, input_boxes, input_boxes_labels, input_points, input_points_labels,
        )
        prompt_features = np.concatenate([text_features, geo_features], axis=1).astype(np.float32)
        prompt_mask = np.concatenate([text_mask.astype(bool), geo_mask.astype(bool)], axis=1)
    else:
        prompt_features = text_features.astype(np.float32)
        prompt_mask = text_mask.astype(bool)

    decoder_outputs, t_dec = model.run_prompt_decoder(vision_features, prompt_features, prompt_mask)
    _, t_post = model.run_postprocess(decoder_outputs, img_size)
    return TimingResult(preprocess_ms=t_pre, vision_encoder_ms=t_vis, text_encoder_ms=t_txt,
                        geometry_encoder_ms=t_geo, decoder_ms=t_dec, postprocess_ms=t_post)


def profile_cached_vision(
    model: ProfileModel, image: torch.Tensor, text: str,
    bbox: np.ndarray | None = None, iterations: int = 20,
) -> list[TimingResult]:
    """Benchmark decoder-only with cached vision & text features."""
    img_size = image.shape[-2:]
    pixel_np, original_sizes, _ = model.run_preprocess(image)
    vision_features, _ = model.run_vision_encoder(pixel_np)
    text_features, text_mask, _ = model.run_text_encoder(text)

    has_geometry = bbox is not None
    if has_geometry:
        input_boxes, input_boxes_labels = _real_box(model.prompt_preprocessor, bbox, original_sizes)
        input_points, input_points_labels = _sentinel_points()

    results: list[TimingResult] = []
    for _ in range(iterations):
        t_geo = 0.0
        if has_geometry:
            geo_features, geo_mask, t_geo = model.run_geometry_encoder(
                vision_features, input_boxes, input_boxes_labels, input_points, input_points_labels,
            )
            prompt_features = np.concatenate([text_features, geo_features], axis=1).astype(np.float32)
            prompt_mask = np.concatenate([text_mask.astype(bool), geo_mask.astype(bool)], axis=1)
        else:
            prompt_features = text_features.astype(np.float32)
            prompt_mask = text_mask.astype(bool)
        decoder_outputs, t_dec = model.run_prompt_decoder(vision_features, prompt_features, prompt_mask)
        _, t_post = model.run_postprocess(decoder_outputs, img_size)
        results.append(TimingResult(geometry_encoder_ms=t_geo, decoder_ms=t_dec, postprocess_ms=t_post))
    return results


def profile_live_stream(
    model: ProfileModel, images: list[torch.Tensor], text: str, iterations: int = 10,
) -> list[TimingResult]:
    """Benchmark live-stream scenario: same prompt, different images."""
    text_features, text_mask, _ = model.run_text_encoder(text)
    prompt_features = text_features.astype(np.float32)
    prompt_mask = text_mask.astype(bool)

    results: list[TimingResult] = []
    img_cycle = images * ((iterations // len(images)) + 1)
    for i in range(iterations):
        image = img_cycle[i % len(img_cycle)]
        img_size = image.shape[-2:]
        pixel_np, _, t_pre = model.run_preprocess(image)
        vision_features, t_vis = model.run_vision_encoder(pixel_np)
        decoder_outputs, t_dec = model.run_prompt_decoder(vision_features, prompt_features, prompt_mask)
        _, t_post = model.run_postprocess(decoder_outputs, img_size)
        results.append(TimingResult(preprocess_ms=t_pre, vision_encoder_ms=t_vis, decoder_ms=t_dec, postprocess_ms=t_post))
    return results


def _profile_variant_config(
    model: ProfileModel, label: str, config_name: str,
    prompt_configs: list[tuple[str, str, np.ndarray | None]],
    primary_image: torch.Tensor, real_images: list[torch.Tensor],
    warmup: int, iterations: int, live_frames: int,
) -> list[ProfileResult]:
    """Run all profile benchmarks for one variant + config combination."""
    results: list[ProfileResult] = []
    for prompt_label, text, bbox in prompt_configs:
        logger.info("  %s prompt — warmup(%d) + measure(%d)", prompt_label, warmup, iterations)
        for _ in range(warmup):
            profile_single_inference(model, primary_image, text, bbox)
        result = ProfileResult(variant=label, config_name=config_name, prompt_type=prompt_label)
        for _ in range(iterations):
            result.timings.append(profile_single_inference(model, primary_image, text, bbox))
        results.append(result)
        cached_result = ProfileResult(variant=label, config_name=config_name, prompt_type=f"{prompt_label}(cached-vis)")
        cached_result.timings = profile_cached_vision(model, primary_image, text, bbox, iterations=iterations)
        results.append(cached_result)

    logger.info("  live-stream — warmup(%d) + measure(%d)", warmup, live_frames)
    for _ in range(warmup):
        profile_single_inference(model, real_images[0], "elephant", None)
    live_result = ProfileResult(variant=label, config_name=config_name, prompt_type="live-stream")
    live_result.timings = profile_live_stream(model, real_images, "elephant", iterations=live_frames)
    results.append(live_result)
    return results


def run_profile(
    base_dir: Path | None, variants: list[str], device: str = "CPU",
    warmup: int = 3, iterations: int = 10, live_frames: int = 15,
    ov_configs: dict[str, dict] | None = None,
) -> list[ProfileResult]:
    """Run all profile benchmarks across variants, prompt types, and configs."""
    coco_dir = Path("examples/assets/coco")
    real_images: list[torch.Tensor] = []
    for img_file in sorted(coco_dir.glob("*.jpg"))[:4]:
        img = _profile_make_real_image(img_file)
        if img is not None:
            real_images.append(img)
    if not real_images:
        logger.warning("No real images found, using dummy images")
        real_images = [_profile_make_dummy_image() for _ in range(3)]

    primary_image = real_images[0]
    sample_bbox = np.array([[216, 184, 458, 436]])
    pc = np.array([337, 310])
    point_bbox = np.array([[pc[0] - 10, pc[1] - 10, pc[0] + 10, pc[1] + 10]])

    prompt_configs: list[tuple[str, str, np.ndarray | None]] = [
        ("text", "elephant", None),
        ("box", "visual", sample_bbox),
        ("point-as-box", "visual", point_bbox),
    ]

    all_results: list[ProfileResult] = []
    for variant_dir_name in variants:
        variant_path = _resolve_variant_path(base_dir, variant_dir_name)
        if variant_path is None:
            continue
        label = _PROFILE_DEFAULT_VARIANTS.get(variant_dir_name, variant_dir_name)
        configs = ov_configs if ov_configs is not None else _get_ov_configs(device)
        for config_name, config in configs.items():
            logger.info("=" * 60)
            logger.info("%s — %s", label, config_name)
            logger.info("=" * 60)
            try:
                model = ProfileModel(variant_path, device, config)
            except Exception:
                logger.exception("Failed to load %s with config %s", variant_dir_name, config_name)
                continue
            try:
                all_results.extend(_profile_variant_config(
                    model=model, label=label, config_name=config_name,
                    prompt_configs=prompt_configs, primary_image=primary_image,
                    real_images=real_images, warmup=warmup, iterations=iterations,
                    live_frames=live_frames,
                ))
            except RuntimeError as exc:
                logger.error("Runtime error for %s/%s: %s", label, config_name, exc)
                continue
            del model
            gc.collect()
    return all_results


def _get_device_full_name(device: str) -> str:
    """Get the full device name as reported by OpenVINO."""
    import openvino as ov  # noqa: PLC0415

    try:
        return ov.Core().get_property(device, "FULL_DEVICE_NAME")
    except RuntimeError:
        return device


def print_profile_summary(results: list[ProfileResult]) -> None:
    """Print a rich summary table of profile results."""
    from rich.console import Console  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415

    con = Console()
    table = Table(title="SAM3 OpenVINO Profile (ms)", show_header=True, header_style="bold cyan")
    for col in ["Variant", "Config", "Prompt", "Preproc", "Vision", "Text", "Geo", "Decoder", "Post", "Total", "FPS"]:
        justify = "right" if col not in {"Variant", "Config", "Prompt"} else "left"
        style = "bold" if col == "Total" else ("magenta" if col == "FPS" else None)
        table.add_column(col, justify=justify, style=style)
    for r in results:
        total = r.stats("total_ms")["mean"]
        fps = 1000.0 / total if total > 0 else 0
        table.add_row(
            r.variant, r.config_name, r.prompt_type,
            f"{r.stats('preprocess_ms')['mean']:.1f}", f"{r.stats('vision_encoder_ms')['mean']:.1f}",
            f"{r.stats('text_encoder_ms')['mean']:.1f}", f"{r.stats('geometry_encoder_ms')['mean']:.1f}",
            f"{r.stats('decoder_ms')['mean']:.1f}", f"{r.stats('postprocess_ms')['mean']:.1f}",
            f"{total:.1f}", f"{fps:.2f}",
        )
    con.print(table)


def print_profile_speedup(results: list[ProfileResult]) -> None:
    """Print speedup of quantized variants relative to FP16."""
    from rich.console import Console  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415

    default_text = [r for r in results if r.config_name == "default" and r.prompt_type == "text"]
    fp16 = [r for r in default_text if "FP16" in r.variant]
    if not fp16:
        return
    baseline = fp16[0].stats("total_ms")["mean"]
    con = Console()
    table = Table(title="Speedup vs OV-FP16 (text prompt)", show_header=True)
    table.add_column("Variant", style="cyan")
    table.add_column("Total (ms)", justify="right")
    table.add_column("Speedup", justify="right", style="green")
    for r in default_text:
        total = r.stats("total_ms")["mean"]
        table.add_row(r.variant, f"{total:.1f}", f"{baseline / total:.2f}x" if total > 0 else "—")
    con.print(table)


def print_profile_live_fps(results: list[ProfileResult]) -> None:
    """Print live-stream FPS comparison."""
    from rich.console import Console  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415

    live = [r for r in results if r.prompt_type == "live-stream"]
    if not live:
        return
    con = Console()
    table = Table(title="Live Stream FPS", show_header=True)
    table.add_column("Variant", style="cyan")
    table.add_column("Config")
    table.add_column("Total (ms)", justify="right")
    table.add_column("FPS", justify="right", style="bold magenta")
    for r in live:
        total = r.stats("total_ms")["mean"]
        table.add_row(r.variant, r.config_name, f"{total:.1f}", f"{1000.0 / total:.1f}" if total > 0 else "—")
    con.print(table)


def save_profile_results(results: list[ProfileResult], device: str, output_dir: Path) -> Path:
    """Save profile results to an Excel file."""
    import openvino as ov  # noqa: PLC0415

    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError as exc:
        msg = "pandas and openpyxl required. Install with: uv pip install pandas openpyxl"
        raise ImportError(msg) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = _get_device_full_name(device)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"sam3_profile_{device.lower()}_{timestamp}.xlsx"

    components = [
        ("preprocess_ms", "Preprocess"), ("vision_encoder_ms", "Vision Encoder"),
        ("text_encoder_ms", "Text Encoder"), ("geometry_encoder_ms", "Geometry Encoder"),
        ("decoder_ms", "Prompt Decoder"), ("postprocess_ms", "Postprocess"),
        ("total_ms", "Total"), ("without_vision_ms", "w/o Vision"),
    ]
    rows: list[dict] = []
    for r in results:
        row: dict = {"Variant": r.variant, "Config": r.config_name, "Prompt": r.prompt_type, "Iterations": len(r.timings)}
        total = r.stats("total_ms")["mean"]
        row["FPS"] = round(1000.0 / total, 2) if total > 0 else 0
        for attr, label in components:
            s = r.stats(attr)
            for sn in ("mean", "median", "std", "min", "max"):
                row[f"{label} {sn.title()} (ms)"] = round(s[sn], 2)
        rows.append(row)

    meta_df = pd.DataFrame([
        {"Key": "Device", "Value": device}, {"Key": "Device Full Name", "Value": device_name},
        {"Key": "Timestamp (UTC)", "Value": timestamp}, {"Key": "OpenVINO Version", "Value": ov.get_version()},
    ])
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        meta_df.to_excel(writer, sheet_name="Device Info", index=False)
        pd.DataFrame(rows).to_excel(writer, sheet_name="Profile Results", index=False)
    logger.info("Profile results saved to %s", filepath)
    return filepath


def save_profile_charts(results: list[ProfileResult], device: str, output_dir: Path) -> Path | None:
    """Generate profile comparison charts and save as PNG."""
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        logger.warning("matplotlib not installed — skipping charts")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = _get_device_full_name(device)

    default_full = [r for r in results if r.config_name == "default" and "cached" not in r.prompt_type and r.prompt_type != "live-stream"]
    live_results = [r for r in results if r.prompt_type == "live-stream" and r.config_name == "default"]
    if not default_full:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"SAM3 OpenVINO Profile — {device_name}", fontsize=14, fontweight="bold")

    # Panel 1: End-to-end latency
    ax1 = axes[0]
    variants = sorted({r.variant for r in default_full})
    prompt_types = sorted({r.prompt_type for r in default_full})
    bw = 0.8 / max(len(prompt_types), 1)
    x = np.arange(len(variants))
    for i, pt in enumerate(prompt_types):
        means = [next((r.stats("total_ms")["mean"] for r in default_full if r.variant == v and r.prompt_type == pt), 0) for v in variants]
        ax1.bar(x + i * bw, means, bw, label=pt)
    ax1.set_xlabel("Variant"); ax1.set_ylabel("Latency (ms)"); ax1.set_title("End-to-End Latency")
    ax1.set_xticks(x + bw * (len(prompt_types) - 1) / 2); ax1.set_xticklabels(variants, rotation=15, ha="right", fontsize=9)
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    # Panel 2: Component breakdown (text)
    ax2 = axes[1]
    text_results = [r for r in default_full if r.prompt_type == "text"]
    comp_attrs = [("preprocess_ms", "Preprocess"), ("vision_encoder_ms", "Vision"), ("text_encoder_ms", "Text"),
                  ("geometry_encoder_ms", "Geometry"), ("decoder_ms", "Decoder"), ("postprocess_ms", "Postprocess")]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#64b5cd"]
    tv = [r.variant for r in text_results]; x2 = np.arange(len(tv)); bottom = np.zeros(len(tv))
    for (attr, label), color in zip(comp_attrs, colors, strict=False):
        vals = [r.stats(attr)["mean"] for r in text_results]
        ax2.bar(x2, vals, bottom=bottom, label=label, color=color, width=0.6); bottom += np.array(vals)
    ax2.set_xlabel("Variant"); ax2.set_ylabel("Latency (ms)"); ax2.set_title("Component Breakdown (text)")
    ax2.set_xticks(x2); ax2.set_xticklabels(tv, rotation=15, ha="right", fontsize=9)
    ax2.legend(fontsize=7, loc="upper left"); ax2.grid(axis="y", alpha=0.3)

    # Panel 3: Live FPS
    ax3 = axes[2]
    if live_results:
        lv = [r.variant for r in live_results]
        fps_vals = [1000.0 / r.stats("total_ms")["mean"] if r.stats("total_ms")["mean"] > 0 else 0 for r in live_results]
        x3 = np.arange(len(lv)); bars = ax3.bar(x3, fps_vals, color="#55a868", width=0.6)
        for bar, fps in zip(bars, fps_vals, strict=False):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{fps:.1f}", ha="center", va="bottom", fontsize=9)
        ax3.set_xlabel("Variant"); ax3.set_ylabel("FPS"); ax3.set_title("Live Stream FPS")
        ax3.set_xticks(x3); ax3.set_xticklabels(lv, rotation=15, ha="right", fontsize=9); ax3.grid(axis="y", alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No live-stream data", transform=ax3.transAxes, ha="center", va="center")

    plt.tight_layout()
    chart_path = output_dir / f"sam3_profile_{device.lower()}_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Chart saved to %s", chart_path)
    return chart_path


# ===========================================================================
# CLI with subcommands
# ===========================================================================


def _build_accuracy_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the 'accuracy' subcommand."""
    p = subparsers.add_parser("accuracy", help="Benchmark accuracy + latency on COCO datasets (PyTorch & OV).")
    p.add_argument("--backend", choices=["openvino", "pytorch"], default="openvino")
    p.add_argument("--variants", nargs="+", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--modes", nargs="+", choices=["text", "visual_exemplar"], default=["text", "visual_exemplar"])
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--warmup", type=int, default=N_WARMUP)
    p.add_argument("--output", type=Path, default=None)


def _build_profile_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the 'profile' subcommand."""
    p = subparsers.add_parser("profile", help="Component-level OV latency profiling with charts and Excel.")
    p.add_argument("--base-dir", type=Path, default=None, help="Local model dir. If omitted, downloads from HF.")
    p.add_argument("--variants", nargs="+", default=list(_PROFILE_DEFAULT_VARIANTS.keys()))
    p.add_argument("--device", type=str, default="CPU")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--live-frames", type=int, default=15)
    p.add_argument("--detail", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("./benchmark_results"))
    p.add_argument("--no-charts", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=None)


def main() -> None:
    """Unified SAM3 benchmark CLI with accuracy and profile subcommands."""
    parser = argparse.ArgumentParser(description="SAM3 benchmark: accuracy evaluation and component-level profiling.")
    subparsers = parser.add_subparsers(dest="command", help="Benchmark mode")
    _build_accuracy_parser(subparsers)
    _build_profile_parser(subparsers)
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "accuracy":
        _run_accuracy(args)
    elif args.command == "profile":
        _run_profile(args)


def _run_accuracy(args: argparse.Namespace) -> None:
    """Execute the accuracy subcommand."""
    global N_WARMUP  # noqa: PLW0603
    N_WARMUP = args.warmup

    if args.datasets:
        datasets = {}
        for ds_name in args.datasets:
            if ds_name in DEFAULT_DATASETS:
                datasets[ds_name] = DEFAULT_DATASETS[ds_name]
            else:
                ann_path = args.data_root / ds_name / "annotations" / "instances_default.json"
                if ann_path.exists():
                    with ann_path.open(encoding="utf-8") as f:
                        datasets[ds_name] = [c["name"] for c in json.load(f)["categories"]]
                else:
                    logger.warning("Dataset %s not found at %s", ds_name, ann_path)
    else:
        datasets = DEFAULT_DATASETS

    variants = ["pytorch"] if args.backend == "pytorch" else (args.variants or ALL_OV_VARIANTS)

    logger.info("=" * 70)
    logger.info("SAM3 Accuracy Benchmark")
    logger.info("  Backend: %s | Variants: %s | Device: %s", args.backend, variants, args.device)
    logger.info("  Modes: %s | Datasets: %s | Warmup: %d", args.modes, list(datasets.keys()), N_WARMUP)
    logger.info("=" * 70)

    results = run_benchmark(backend=args.backend, variants=variants, device=args.device,
                            data_root=args.data_root, datasets=datasets, modes=args.modes)
    print_results_table(results)

    output_path = args.output or Path(f"benchmark_results_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
    save_results_json(results, output_path)


def _run_profile(args: argparse.Namespace) -> None:
    """Execute the profile subcommand."""
    device_name = _get_device_full_name(args.device)
    logger.info("=" * 70)
    logger.info("SAM3 OpenVINO Profile")
    logger.info("  Device: %s (%s)", args.device, device_name)
    logger.info("  Variants: %s", args.variants)
    logger.info("  Warmup: %d | Iterations: %d | Live frames: %d", args.warmup, args.iterations, args.live_frames)
    logger.info("=" * 70)

    cache_dir: Path | None = args.cache_dir
    if cache_dir is None and args.device.upper().startswith("GPU"):
        cache_dir = Path("./ov_cache")
        logger.info("GPU detected — enabling model cache: %s", cache_dir)

    ov_configs = _get_ov_configs(args.device, cache_dir=cache_dir)
    results = run_profile(base_dir=args.base_dir, variants=args.variants, device=args.device,
                          warmup=args.warmup, iterations=args.iterations, live_frames=args.live_frames,
                          ov_configs=ov_configs)

    if not results:
        logger.error("No profile results collected.")
        return

    print_profile_summary(results)
    print_profile_speedup(results)
    print_profile_live_fps(results)

    if args.detail:
        from rich.console import Console as RichConsole  # noqa: PLC0415
        from rich.table import Table as RichTable  # noqa: PLC0415
        rc = RichConsole()
        for r in results:
            rc.print(f"\n[bold]{r.variant} | {r.config_name} | {r.prompt_type}[/bold]")
            dt = RichTable(show_header=True, box=None, pad_edge=False)
            dt.add_column("Component", width=18)
            for col in ("Mean", "Median", "Std", "Min", "Max"):
                dt.add_column(col, justify="right")
            for attr, label in [("preprocess_ms", "Preprocess"), ("vision_encoder_ms", "Vision Encoder"),
                                ("text_encoder_ms", "Text Encoder"), ("geometry_encoder_ms", "Geometry Encoder"),
                                ("decoder_ms", "Prompt Decoder"), ("postprocess_ms", "Postprocess"),
                                ("total_ms", "TOTAL"), ("without_vision_ms", "w/o Vision")]:
                s = r.stats(attr)
                dt.add_row(label, f"{s['mean']:.1f}", f"{s['median']:.1f}", f"{s['std']:.1f}",
                           f"{s['min']:.1f}", f"{s['max']:.1f}", style="bold" if attr == "total_ms" else "")
            rc.print(dt)

    save_profile_results(results, args.device, args.output_dir)
    if not args.no_charts:
        save_profile_charts(results, args.device, args.output_dir)


if __name__ == "__main__":
    main()
