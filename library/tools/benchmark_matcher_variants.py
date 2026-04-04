# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmark SAM-HQ decoder x DINOv3 encoder variants: export, GPU/CPU inference, accuracy, determinism.

Usage:
    cd library
    uv run python tools/benchmark_matcher_variants.py
    uv run python tools/benchmark_matcher_variants.py --sam SAM-HQ-base SAM-HQ-large
    uv run python tools/benchmark_matcher_variants.py --encoder dinov3_small dinov3_base
    uv run python tools/benchmark_matcher_variants.py --sam SAM-HQ-base --encoder dinov3_small dinov3_large
    uv run python tools/benchmark_matcher_variants.py --sam SAM-HQ-tiny --gpu-iterations 20
"""

import argparse
import logging
import shutil
from pathlib import Path
from time import time

import numpy as np
import openvino
import torch
import torch.nn.functional as F  # noqa: N812

from instantlearn.data import Sample
from instantlearn.models import Matcher
from instantlearn.utils.constants import SAMModelName

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# All SAM-HQ variants to benchmark
ALL_SAM_HQ_VARIANTS = [
    SAMModelName.SAM_HQ_TINY,
    SAMModelName.SAM_HQ_BASE,
    SAMModelName.SAM_HQ_LARGE,
    SAMModelName.SAM_HQ,
]

# All DINOv3 encoder variants
ALL_ENCODERS = [
    "dinov3_small",
    "dinov3_base",
    "dinov3_large",
]

ROOT_DIR = Path("examples/assets/coco")
REF_SAMPLE = Sample(
    image_path=str(ROOT_DIR / "000000286874.jpg"),
    mask_paths=str(ROOT_DIR / "000000286874_mask.png"),
)
TARGET_SAMPLE = Sample(image_path=str(ROOT_DIR / "000000173279.jpg"))


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two binary mask arrays."""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def benchmark_variant(
    variant: SAMModelName,
    device: str,
    encoder: str = "dinov3_large",
    gpu_iterations: int = 10,
    compress_fp16: bool = True,
) -> dict:
    """Benchmark a single SAM decoder + encoder combination end-to-end."""
    combo_name = f"{variant} + {encoder}"
    result = {
        "variant": combo_name,
        "status": "success",
        "pytorch_time": None,
        "export_time": None,
        "ov_gpu_time_first": None,
        "ov_gpu_times": [],
        "ov_cpu_time": None,
        "model_size_mb": None,
        "pt_scores": None,
        "ov_gpu_scores": None,
        "ov_cpu_scores": None,
        "gpu_score_std": None,
        "gpu_mask_iou_vs_pt": None,
        "cpu_mask_iou_vs_pt": None,
        "gpu_cpu_mask_iou": None,
        "gpu_mask_iou_consistency": None,
    }

    try:
        # --- 1. PyTorch inference ---
        logger.info(f"[{combo_name}] Loading model...")
        model = Matcher(device=device, sam=variant, encoder_model=encoder)
        model.fit(REF_SAMPLE)

        tic = time()
        predictions = model.predict(TARGET_SAMPLE)
        result["pytorch_time"] = time() - tic
        pt_masks = predictions[0]["pred_masks"].cpu().numpy()
        result["pt_scores"] = predictions[0]["pred_scores"].cpu().numpy().round(4).tolist()
        logger.info(f"[{combo_name}] PyTorch: {result['pytorch_time']:.3f}s, scores={result['pt_scores']}")

        # --- 2. Export to OpenVINO ---
        safe_name = f"{variant}_{encoder}".replace(" ", "_").replace("-", "_")
        export_dir = Path(f"./exports/benchmark_{safe_name}")
        if export_dir.exists():
            shutil.rmtree(export_dir)

        tic = time()
        ov_path = model.export(
            export_dir=export_dir,
            backend="openvino",
            compress_to_fp16=compress_fp16,
        )
        result["export_time"] = time() - tic
        logger.info(f"[{combo_name}] Export: {result['export_time']:.1f}s → {ov_path}")

        # Model size
        total_size = sum(f.stat().st_size for f in export_dir.rglob("*") if f.is_file())
        result["model_size_mb"] = round(total_size / (1024 * 1024), 1)
        logger.info(f"[{combo_name}] Model size: {result['model_size_mb']} MB")

        # Free PyTorch model
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        # --- 3. OpenVINO inference ---
        core = openvino.Core()
        ov_model = core.read_model(str(ov_path))

        # Prepare input (add batch dimension for NCHW layout)
        input_data = TARGET_SAMPLE.image.numpy()[None]  # [1, 3, H, W]
        expected_shape = tuple(ov_model.input(0).shape)
        if input_data.shape != expected_shape:
            tensor = torch.from_numpy(input_data)
            tensor = F.interpolate(tensor, size=(expected_shape[2], expected_shape[3]), mode="bilinear")
            input_data = tensor.numpy()

        # --- GPU inference ---
        compiled_gpu = core.compile_model(
            ov_model,
            "GPU",
            config={
                "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": "1",
            },
        )

        # Warmup
        compiled_gpu(input_data)

        pt_h, pt_w = pt_masks.shape[-2], pt_masks.shape[-1]

        def resize_masks_to_pt(masks: np.ndarray) -> np.ndarray:
            """Resize OV masks to match PyTorch output resolution for IoU comparison."""
            if masks.shape[-2:] == (pt_h, pt_w):
                return masks
            t = torch.from_numpy(masks).unsqueeze(0).float()
            t = F.interpolate(t, size=(pt_h, pt_w), mode="bilinear")
            return t.squeeze(0).numpy() > 0.5

        # First timed run
        tic = time()
        outputs = compiled_gpu(input_data)
        result["ov_gpu_time_first"] = time() - tic
        masks_port = compiled_gpu.output("masks")
        scores_port = compiled_gpu.output("scores")
        ov_gpu_masks = np.asarray(outputs[masks_port])
        ov_gpu_scores = np.asarray(outputs[scores_port])
        result["ov_gpu_scores"] = ov_gpu_scores.round(4).tolist()

        # GPU vs PyTorch mask IoU (resize OV masks to PT resolution)
        result["gpu_mask_iou_vs_pt"] = round(compute_iou(pt_masks, resize_masks_to_pt(ov_gpu_masks)), 4)

        # Determinism test: N GPU iterations
        all_gpu_scores = [ov_gpu_scores.copy()]
        all_gpu_masks = [ov_gpu_masks.copy()]
        for _ in range(gpu_iterations - 1):
            tic = time()
            outputs = compiled_gpu(input_data)
            result["ov_gpu_times"].append(time() - tic)
            all_gpu_masks.append(np.asarray(outputs[masks_port]).copy())
            all_gpu_scores.append(np.asarray(outputs[scores_port]).copy())

        # Score std across GPU runs
        scores_array = np.array(all_gpu_scores)
        result["gpu_score_std"] = round(float(scores_array.std(axis=0).mean()), 6)

        # Mask IoU consistency across GPU runs
        ious = [compute_iou(all_gpu_masks[0], all_gpu_masks[i]) for i in range(1, len(all_gpu_masks))]
        result["gpu_mask_iou_consistency"] = round(float(np.mean(ious)), 4) if ious else 1.0

        avg_gpu_time = np.mean(result["ov_gpu_times"]) if result["ov_gpu_times"] else result["ov_gpu_time_first"]
        logger.info(
            f"[{combo_name}] GPU: first={result['ov_gpu_time_first']:.3f}s, "
            f"avg={avg_gpu_time:.3f}s, score_std={result['gpu_score_std']:.6f}, "
            f"mask_iou_consistency={result['gpu_mask_iou_consistency']:.4f}, "
            f"mask_iou_vs_pt={result['gpu_mask_iou_vs_pt']:.4f}",
        )

        del compiled_gpu

        # --- CPU inference ---
        compiled_cpu = core.compile_model(ov_model, "CPU")
        # Warmup
        compiled_cpu(input_data)

        tic = time()
        outputs = compiled_cpu(input_data)
        result["ov_cpu_time"] = time() - tic
        cpu_masks_port = compiled_cpu.output("masks")
        cpu_scores_port = compiled_cpu.output("scores")
        ov_cpu_masks = np.asarray(outputs[cpu_masks_port])
        ov_cpu_scores = np.asarray(outputs[cpu_scores_port])
        result["ov_cpu_scores"] = ov_cpu_scores.round(4).tolist()
        result["cpu_mask_iou_vs_pt"] = round(compute_iou(pt_masks, resize_masks_to_pt(ov_cpu_masks)), 4)
        result["gpu_cpu_mask_iou"] = round(compute_iou(ov_gpu_masks, ov_cpu_masks), 4)
        logger.info(
            f"[{combo_name}] CPU: {result['ov_cpu_time']:.3f}s, "
            f"mask_iou_vs_pt={result['cpu_mask_iou_vs_pt']:.4f}, "
            f"gpu_cpu_iou={result['gpu_cpu_mask_iou']:.4f}",
        )

        del compiled_cpu

    except Exception:
        result["status"] = "FAILED"
        logger.exception(f"[{combo_name}] Failed")

    return result


def print_summary(results: list[dict]) -> None:
    """Print a summary table of all benchmark results."""
    print("\n" + "=" * 120)
    print("BENCHMARK SUMMARY")
    print("=" * 120)

    header = (
        f"{'Variant':<36} {'Status':<8} {'Size(MB)':>9} {'PT(ms)':>8} {'GPU(ms)':>8} "
        f"{'CPU(ms)':>8} {'ScoreStd':>9} {'GPU↔PT':>7} {'CPU↔PT':>7} {'GPU↔CPU':>8} {'GPU cons':>9}"
    )
    print(header)
    print("-" * 140)

    for r in results:
        if r["status"] == "FAILED":
            print(f"{r['variant']:<36} {'FAILED':<8}")
            continue

        gpu_avg = np.mean(r["ov_gpu_times"]) * 1000 if r["ov_gpu_times"] else (r["ov_gpu_time_first"] or 0) * 1000
        print(
            f"{r['variant']:<36} {'OK':<8} "
            f"{r['model_size_mb'] or 0:>9.1f} "
            f"{(r['pytorch_time'] or 0) * 1000:>8.0f} "
            f"{gpu_avg:>8.1f} "
            f"{(r['ov_cpu_time'] or 0) * 1000:>8.0f} "
            f"{r['gpu_score_std'] or 0:>9.6f} "
            f"{r['gpu_mask_iou_vs_pt'] or 0:>7.4f} "
            f"{r['cpu_mask_iou_vs_pt'] or 0:>7.4f} "
            f"{r['gpu_cpu_mask_iou'] or 0:>8.4f} "
            f"{r['gpu_mask_iou_consistency'] or 0:>9.4f}",
        )

    print("=" * 140)
    print("Columns: Size=model files, PT=PyTorch, GPU/CPU=OpenVINO avg latency")
    print("ScoreStd=score std across GPU runs, GPU↔PT/CPU↔PT=mask IoU vs PyTorch")
    print("GPU↔CPU=mask IoU between GPU and CPU, GPU cons=mask IoU consistency across GPU runs")


def main() -> None:
    """Run SAM-HQ decoder x DINOv3 encoder benchmark across selected variants."""
    parser = argparse.ArgumentParser(description="Benchmark SAM-HQ decoder x DINOv3 encoder variants")
    parser.add_argument(
        "--sam",
        nargs="+",
        default=None,
        help="SAM variants to test (e.g., SAM-HQ-tiny SAM-HQ-base). Default: all SAM-HQ variants.",
    )
    parser.add_argument(
        "--encoder",
        nargs="+",
        default=None,
        help="DINOv3 encoder variants (e.g., dinov3_small dinov3_large). Default: all encoders.",
    )
    parser.add_argument("--gpu-iterations", type=int, default=10, help="GPU determinism iterations (default: 10)")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 compression")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    sam_variants = [SAMModelName(v) for v in args.sam] if args.sam else ALL_SAM_HQ_VARIANTS
    encoders = args.encoder or ALL_ENCODERS

    combos = [(sam, enc) for sam in sam_variants for enc in encoders]
    logger.info("Testing %d combinations: %d SAM x %d encoders", len(combos), len(sam_variants), len(encoders))

    results = []
    for sam_variant, encoder in combos:
        logger.info(f"\n{'=' * 60}\nBenchmarking: {sam_variant} + {encoder}\n{'=' * 60}")
        result = benchmark_variant(
            sam_variant,
            device=device,
            encoder=encoder,
            gpu_iterations=args.gpu_iterations,
            compress_fp16=not args.no_fp16,
        )
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()
