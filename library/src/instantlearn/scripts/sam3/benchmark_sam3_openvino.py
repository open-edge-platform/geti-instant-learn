# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmark SAM3 OpenVINO inference across model variants, prompt types, and optimizations.

Measures component-level and end-to-end latency for live-inference scenarios:

**Model variants benchmarked:**
  - OpenVINO IR: FP16, NNCF-INT8, NNCF-INT4

**Prompt types:**
  - Text prompt (category name → detect all instances)
  - Box prompt (bounding box → segment specific ROI)
  - Point-as-box prompt (point expanded to small box)

**Scenarios:**
  - Cold start: Full pipeline including vision encoder
  - Warm (cached vision features): Same image, new prompts — decoder only
  - Live stream: Same prompt, different images — measures per-frame throughput

**Optimizations tested:**
  - Default OpenVINO settings
  - LATENCY performance hint
  - Reduced inference threads
  - BF16 precision hint (if supported)

**Device support:**
  - CPU (default)
  - GPU — Intel iGPU/dGPU (pass ``--device GPU``). XPU users should use
    ``--device GPU`` since OpenVINO maps Intel XPU as GPU device.
  - AUTO — let OpenVINO select the best available device

**Component breakdown:**
  - Image preprocessing (CPU/PyTorch)
  - Vision encoder (OpenVINO)
  - Text encoder (OpenVINO)
  - Geometry encoder (OpenVINO) — for box/point prompts
  - Prompt decoder (OpenVINO)
  - Postprocessing (CPU/PyTorch)

**Output:**
  - Rich terminal tables with summary, variant comparison, speedups, and FPS
  - Excel file with detailed results (``sam3_benchmark_{device}_{timestamp}.xlsx``)
  - PNG chart with latency comparison, component breakdown, and FPS
    (requires matplotlib; use ``--no-charts`` to skip)

Usage:
    # Auto-download models from HuggingFace and benchmark (default)
    python scripts/benchmark_sam3_openvino.py

    # Auto-download INT8 quantised variant
    python scripts/benchmark_sam3_openvino.py --variants openvino-int8

    # Use local model directory instead of HuggingFace
    python scripts/benchmark_sam3_openvino.py --base-dir ./sam3-openvino

    # Benchmark on Intel GPU (XPU)
    python scripts/benchmark_sam3_openvino.py --device GPU

    # More warmup / iterations
    python scripts/benchmark_sam3_openvino.py --warmup 5 --iterations 20
"""

from __future__ import annotations

import argparse
import gc
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.table import Table
from transformers import CLIPTokenizerFast

from instantlearn.models.sam3.processing import (
    Sam3Postprocessor,
    Sam3Preprocessor,
    Sam3PromptPreprocessor,
)

logger = logging.getLogger(__name__)
console = Console()

RESOLUTION = 1008

# Default HuggingFace repo containing exported SAM3 OpenVINO models
SAM3_HF_REPO_ID = "rajeshgangireddy/exported_sam3"

# Canonical model file names (v3 five-model split)
VISION_ENCODER = "vision-encoder"
TEXT_ENCODER = "text-encoder"
GEOMETRY_ENCODER = "geometry-encoder"
GEOMETRY_ENCODER_EXEMPLAR = "geometry-encoder-exemplar"
PROMPT_DECODER = "prompt-decoder"

# Model variants to benchmark (directory name -> human label)
DEFAULT_VARIANTS: dict[str, str] = {
    "openvino-fp16": "OV-FP16",
    "openvino-nncf-int8": "OV-NNCF-INT8",
    "openvino-nncf-int4": "OV-NNCF-INT4",
}

# OpenVINO compile configs to benchmark (keyed by device category)
OV_CONFIGS_CPU: dict[str, dict] = {
    "default": {},
    "latency-hint": {"PERFORMANCE_HINT": "LATENCY"},
    "throughput-hint": {"PERFORMANCE_HINT": "THROUGHPUT"},
}

# GPU configs — skip throughput-hint (multiplies VRAM usage, causes OOM)
# Enable model caching and FP16 precision hint to stabilise GPU compilation
# on Intel Arc dGPUs (e.g. B580) where the OpenCL JIT compiler can crash
# without these settings.
OV_CONFIGS_GPU: dict[str, dict] = {
    "default": {"INFERENCE_PRECISION_HINT": "f16"},
    "latency-hint": {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f16"},
}


def _is_gpu_available() -> bool:
    """Check whether an OpenVINO GPU device is available."""
    try:
        core = ov.Core()
        return "GPU" in core.available_devices
    except Exception:  # noqa: BKA001
        return False


def _get_ov_configs(device: str, cache_dir: Path | None = None) -> dict[str, dict]:
    """Return device-appropriate compile configs.

    Args:
        device: OpenVINO device string.
        cache_dir: Optional model cache directory. Strongly recommended for GPU
            to avoid repeated OpenCL JIT compilation which can crash on some
            Intel Arc dGPUs.
    """
    base = OV_CONFIGS_GPU if device.upper().startswith("GPU") else OV_CONFIGS_CPU
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        base = {name: {**cfg, "CACHE_DIR": str(cache_dir)} for name, cfg in base.items()}
    return base


def _download_variant(variant: str, repo_id: str = SAM3_HF_REPO_ID) -> Path:
    """Download a model variant from HuggingFace Hub and return its local path.

    Args:
        variant: Variant subdirectory name (e.g. ``openvino-fp16``).
        repo_id: HuggingFace repository ID.

    Returns:
        Local path to the downloaded variant directory.
    """
    console.print(f"  Downloading [cyan]{variant}[/cyan] from [blue]{repo_id}[/blue]...")
    cache_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{variant}/*", "tokenizer*", "special_tokens_map*"],
    )
    return Path(cache_dir) / variant


def _resolve_variant_path(base_dir: Path | None, variant: str) -> Path | None:
    """Resolve a variant to a local directory, downloading from HF if needed.

    Args:
        base_dir: Local base directory, or ``None`` for HuggingFace download.
        variant: Variant subdirectory name.

    Returns:
        Local path to the variant directory, or ``None`` if unavailable.
    """
    if base_dir is not None:
        path = base_dir / variant
        if not path.is_dir():
            console.print(f"[yellow]Skipping {variant}: not found in {base_dir}[/yellow]")
            return None
        return path
    try:
        return _download_variant(variant)
    except Exception:
        logger.exception("Failed to download %s from HuggingFace", variant)
        return None



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
            self.preprocess_ms
            + self.vision_encoder_ms
            + self.text_encoder_ms
            + self.geometry_encoder_ms
            + self.decoder_ms
            + self.postprocess_ms
        )

    @property
    def without_vision_ms(self) -> float:
        """Time without vision encoder (cached features scenario)."""
        return self.text_encoder_ms + self.geometry_encoder_ms + self.decoder_ms + self.postprocess_ms


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results for one configuration."""

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



class BenchmarkModel:
    """Lightweight model wrapper for benchmarking individual components."""

    def __init__(
        self,
        model_dir: Path,
        device: str = "CPU",
        config: dict | None = None,
    ) -> None:
        """Initialize benchmark model with compile configuration.

        Args:
            model_dir: Directory containing model files.
            device: OpenVINO device string.
            config: Optional OpenVINO compile properties.
        """
        core = ov.Core()
        compile_config = config or {}

        vision_path = self._find(model_dir, VISION_ENCODER)
        text_path = self._find(model_dir, TEXT_ENCODER)
        geo_path = self._find(model_dir, GEOMETRY_ENCODER)
        geo_ex_path = self._find(model_dir, GEOMETRY_ENCODER_EXEMPLAR)
        decoder_path = self._find(model_dir, PROMPT_DECODER)

        models = [
            ("vision-encoder", vision_path),
            ("text-encoder", text_path),
            ("geometry-encoder", geo_path),
            ("geometry-encoder-exemplar", geo_ex_path),
            ("prompt-decoder", decoder_path),
        ]
        compiled = {}
        for name, path in models:
            console.print(f"    Compiling [cyan]{name}[/cyan] on {device}...", end=" ")
            t0 = time.perf_counter()
            compiled[name] = core.compile_model(path, device, compile_config)
            elapsed = time.perf_counter() - t0
            console.print(f"[green]{elapsed:.1f}s[/green]")

        self.vision_model = compiled["vision-encoder"]
        self.text_model = compiled["text-encoder"]
        self.geometry_model = compiled["geometry-encoder"]
        self.geometry_exemplar_model = compiled["geometry-encoder-exemplar"]
        self.decoder_model = compiled["prompt-decoder"]

        # Create infer requests for sync inference (avoids request creation overhead)
        self.vision_request = self.vision_model.create_infer_request()
        self.text_request = self.text_model.create_infer_request()
        self.geometry_request = self.geometry_model.create_infer_request()
        self.geometry_exemplar_request = self.geometry_exemplar_model.create_infer_request()
        self.decoder_request = self.decoder_model.create_infer_request()

        # Preprocessors
        self.preprocessor = Sam3Preprocessor(target_size=RESOLUTION)
        self.prompt_preprocessor = Sam3PromptPreprocessor(target_size=RESOLUTION)
        self.postprocessor = Sam3Postprocessor(
            target_size=RESOLUTION,
            threshold=0.5,
            mask_threshold=0.5,
        )

        # Tokenizer — load from model_dir if available, else HuggingFace
        if (model_dir / "tokenizer.json").exists():
            self.tokenizer = CLIPTokenizerFast.from_pretrained(str(model_dir))
        else:
            self.tokenizer = CLIPTokenizerFast.from_pretrained(SAM3_HF_REPO_ID)

    @staticmethod
    def _find(model_dir: Path, name: str) -> Path:
        """Find model file (.xml preferred, then .onnx).

        Raises:
            FileNotFoundError: If no matching model file is found.
        """
        for ext in (".xml", ".onnx", "-fp16.onnx"):
            candidate = model_dir / f"{name}{ext}"
            if candidate.exists():
                return candidate
        # glob fallback
        variants = sorted(model_dir.glob(f"{name}*.onnx"))
        if variants:
            return variants[0]
        msg = f"Model '{name}' not found in {model_dir}"
        raise FileNotFoundError(msg)

    def run_preprocess(
        self,
        image: torch.Tensor,
    ) -> tuple[np.ndarray, list[tuple[int, int]], float]:
        """Preprocess image; returns (pixel_values_np, original_sizes, elapsed_ms)."""
        t0 = time.perf_counter()
        image_4d = image.unsqueeze(0) if image.ndim == 3 else image
        with torch.no_grad():
            pixel_values, original_sizes = self.preprocessor(image_4d)
        pixel_np = pixel_values.numpy()
        elapsed = (time.perf_counter() - t0) * 1000
        return pixel_np, original_sizes, elapsed

    def run_vision_encoder(self, pixel_values: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
        """Run vision encoder; returns (features_dict, elapsed_ms)."""
        t0 = time.perf_counter()
        self.vision_request.infer([pixel_values])
        result = {
            "fpn_feat_0": self.vision_request.get_tensor("fpn_feat_0").data,
            "fpn_feat_1": self.vision_request.get_tensor("fpn_feat_1").data,
            "fpn_feat_2": self.vision_request.get_tensor("fpn_feat_2").data,
            "fpn_pos_2": self.vision_request.get_tensor("fpn_pos_2").data,
        }
        elapsed = (time.perf_counter() - t0) * 1000
        return result, elapsed

    def run_text_encoder(
        self,
        text: str,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Run text encoder; returns (text_features, text_mask, elapsed_ms)."""
        tokens = self.tokenizer([text], return_tensors="np", padding=True)
        input_ids = _pad_or_truncate(tokens.input_ids.astype(np.int64), 32)
        attention_mask = _pad_or_truncate(tokens.attention_mask.astype(np.int64), 32)

        t0 = time.perf_counter()
        self.text_request.infer([input_ids, attention_mask])
        text_features = self.text_request.get_tensor("text_features").data
        text_mask = self.text_request.get_tensor("text_mask").data
        elapsed = (time.perf_counter() - t0) * 1000
        return text_features, text_mask, elapsed

    def run_geometry_encoder(
        self,
        vision_features: dict[str, np.ndarray],
        input_boxes: np.ndarray,
        input_boxes_labels: np.ndarray,
        input_points: np.ndarray,
        input_points_labels: np.ndarray,
        *,
        exemplar: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Run geometry encoder; returns (geometry_features, geometry_mask, elapsed_ms)."""
        request = self.geometry_exemplar_request if exemplar else self.geometry_request
        t0 = time.perf_counter()
        request.infer([
            vision_features["fpn_feat_2"],
            vision_features["fpn_pos_2"],
            input_boxes,
            input_boxes_labels,
            input_points,
            input_points_labels,
        ])
        geo_features = request.get_tensor("geometry_features").data
        geo_mask = request.get_tensor("geometry_mask").data
        elapsed = (time.perf_counter() - t0) * 1000
        return geo_features, geo_mask, elapsed

    def run_prompt_decoder(
        self,
        vision_features: dict[str, np.ndarray],
        prompt_features: np.ndarray,
        prompt_mask: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float]:
        """Run prompt decoder; returns (outputs_dict, elapsed_ms)."""
        t0 = time.perf_counter()
        self.decoder_request.infer([
            vision_features["fpn_feat_0"],
            vision_features["fpn_feat_1"],
            vision_features["fpn_feat_2"],
            vision_features["fpn_pos_2"],
            prompt_features,
            prompt_mask,
        ])
        result = {
            "pred_masks": self.decoder_request.get_tensor("pred_masks").data,
            "pred_boxes": self.decoder_request.get_tensor("pred_boxes").data,
            "pred_logits": self.decoder_request.get_tensor("pred_logits").data,
            "presence_logits": self.decoder_request.get_tensor("presence_logits").data,
        }
        elapsed = (time.perf_counter() - t0) * 1000
        return result, elapsed

    def run_postprocess(
        self,
        decoder_outputs: dict[str, np.ndarray],
        img_size: tuple[int, int],
    ) -> tuple[dict, float]:
        """Run postprocessing; returns (results_dict, elapsed_ms)."""
        t0 = time.perf_counter()
        outputs_torch = {k: torch.from_numpy(np.array(v)) for k, v in decoder_outputs.items()}
        with torch.no_grad():
            result = self.postprocessor(outputs_torch, target_sizes=[img_size])
        elapsed = (time.perf_counter() - t0) * 1000
        return result[0], elapsed



def _pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or truncate array to target sequence length."""
    cur = arr.shape[1]
    if cur == target_len:
        return arr
    if cur > target_len:
        return arr[:, :target_len]
    pad = np.zeros((arr.shape[0], target_len - cur), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)


def _make_dummy_image(h: int = 720, w: int = 1280) -> torch.Tensor:
    """Create a realistic-sized dummy image tensor [3, H, W]."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def _make_real_image(image_path: Path) -> torch.Tensor | None:
    """Load a real image, or return None if not found."""
    if not image_path.exists():
        return None
    import cv2  # noqa: PLC0415

    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def _sentinel_box() -> tuple[np.ndarray, np.ndarray]:
    """Return sentinel box inputs (no box prompt)."""
    boxes = np.zeros((1, 1, 4), dtype=np.float32)
    labels = np.full((1, 1), -10, dtype=np.int64)
    return boxes, labels


def _sentinel_points() -> tuple[np.ndarray, np.ndarray]:
    """Return sentinel point inputs (no point prompt)."""
    points = np.zeros((1, 1, 2), dtype=np.float32)
    labels = np.full((1, 1), -10, dtype=np.int64)
    return points, labels


def _real_box(
    preprocessor: Sam3PromptPreprocessor,
    bbox: np.ndarray,
    original_sizes: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a xyxy bbox to the model's normalized cxcywh format."""
    with torch.no_grad():
        box_tensor, _ = preprocessor(original_sizes, input_boxes=bbox)
    boxes = box_tensor.numpy().astype(np.float32)
    labels = np.ones((1, boxes.shape[1]), dtype=np.int64)
    return boxes, labels



def benchmark_single_inference(
    model: BenchmarkModel,
    image: torch.Tensor,
    text: str,
    bbox: np.ndarray | None = None,
) -> TimingResult:
    """Run one complete inference pass and time each component."""
    img_size = image.shape[-2:]

    # 1. Preprocess
    pixel_np, original_sizes, t_pre = model.run_preprocess(image)

    # 2. Vision encoder
    vision_features, t_vis = model.run_vision_encoder(pixel_np)

    # 3. Text encoder
    text_features, text_mask, t_txt = model.run_text_encoder(text)

    # 4. Geometry encoder (box/point prompts) or text-only
    t_geo = 0.0
    if bbox is not None:
        input_boxes, input_boxes_labels = _real_box(
            model.prompt_preprocessor,
            bbox,
            original_sizes,
        )
        input_points, input_points_labels = _sentinel_points()

        geo_features, geo_mask, t_geo = model.run_geometry_encoder(
            vision_features,
            input_boxes,
            input_boxes_labels,
            input_points,
            input_points_labels,
        )

        # Concatenate text + geometry features
        prompt_features = np.concatenate(
            [text_features, geo_features],
            axis=1,
        ).astype(np.float32)
        prompt_mask = np.concatenate(
            [text_mask.astype(bool), geo_mask.astype(bool)],
            axis=1,
        )
    else:
        prompt_features = text_features.astype(np.float32)
        prompt_mask = text_mask.astype(bool)

    # 5. Prompt decoder
    decoder_outputs, t_dec = model.run_prompt_decoder(
        vision_features,
        prompt_features,
        prompt_mask,
    )

    # 6. Postprocess
    _, t_post = model.run_postprocess(decoder_outputs, img_size)

    return TimingResult(
        preprocess_ms=t_pre,
        vision_encoder_ms=t_vis,
        text_encoder_ms=t_txt,
        geometry_encoder_ms=t_geo,
        decoder_ms=t_dec,
        postprocess_ms=t_post,
    )


def benchmark_cached_vision(
    model: BenchmarkModel,
    image: torch.Tensor,
    text: str,
    bbox: np.ndarray | None = None,
    iterations: int = 20,
) -> list[TimingResult]:
    """Benchmark decoder-only inference with cached vision & text features.

    Simulates the scenario where vision encoder output is cached (same
    image) and text encoder output is cached (same prompt). Only geometry
    encoder (if box/point) + prompt decoder + postprocess run per iteration.
    """
    img_size = image.shape[-2:]

    # Pre-compute everything that can be cached
    pixel_np, original_sizes, _ = model.run_preprocess(image)
    vision_features, _ = model.run_vision_encoder(pixel_np)
    text_features, text_mask, _ = model.run_text_encoder(text)

    has_geometry = bbox is not None
    if has_geometry:
        input_boxes, input_boxes_labels = _real_box(
            model.prompt_preprocessor,
            bbox,
            original_sizes,
        )
        input_points, input_points_labels = _sentinel_points()

    results: list[TimingResult] = []
    for _ in range(iterations):
        t_geo = 0.0
        if has_geometry:
            geo_features, geo_mask, t_geo = model.run_geometry_encoder(
                vision_features,
                input_boxes,
                input_boxes_labels,
                input_points,
                input_points_labels,
            )
            prompt_features = np.concatenate(
                [text_features, geo_features],
                axis=1,
            ).astype(np.float32)
            prompt_mask = np.concatenate(
                [text_mask.astype(bool), geo_mask.astype(bool)],
                axis=1,
            )
        else:
            prompt_features = text_features.astype(np.float32)
            prompt_mask = text_mask.astype(bool)

        decoder_outputs, t_dec = model.run_prompt_decoder(
            vision_features,
            prompt_features,
            prompt_mask,
        )
        _, t_post = model.run_postprocess(decoder_outputs, img_size)
        results.append(
            TimingResult(
                geometry_encoder_ms=t_geo,
                decoder_ms=t_dec,
                postprocess_ms=t_post,
            ),
        )
    return results


def benchmark_live_stream(
    model: BenchmarkModel,
    images: list[torch.Tensor],
    text: str,
    iterations: int = 10,
) -> list[TimingResult]:
    """Benchmark live-stream scenario: same prompt, different images.

    Text encoder output is cached; vision encoder runs per frame.
    Text-only prompt — no geometry encoder.
    """
    # Cache text features once
    text_features, text_mask, _ = model.run_text_encoder(text)
    prompt_features = text_features.astype(np.float32)
    prompt_mask = text_mask.astype(bool)

    results: list[TimingResult] = []
    img_cycle = images * ((iterations // len(images)) + 1)

    for i in range(iterations):
        image = img_cycle[i % len(img_cycle)]
        img_size = image.shape[-2:]

        pixel_np, _orig, t_pre = model.run_preprocess(image)
        vision_features, t_vis = model.run_vision_encoder(pixel_np)

        decoder_outputs, t_dec = model.run_prompt_decoder(
            vision_features,
            prompt_features,
            prompt_mask,
        )
        _, t_post = model.run_postprocess(decoder_outputs, img_size)

        results.append(
            TimingResult(
                preprocess_ms=t_pre,
                vision_encoder_ms=t_vis,
                decoder_ms=t_dec,
                postprocess_ms=t_post,
            ),
        )
    return results



def _benchmark_variant_config(
    model: BenchmarkModel,
    label: str,
    config_name: str,
    prompt_configs: list[tuple[str, str, np.ndarray | None]],
    primary_image: torch.Tensor,
    real_images: list[torch.Tensor],
    warmup: int,
    iterations: int,
    live_frames: int,
) -> list[BenchmarkResult]:
    """Run all benchmarks for one variant + config combination."""
    results: list[BenchmarkResult] = []

    for prompt_label, text, bbox in prompt_configs:
        console.print(f"  [green]{prompt_label}[/green] prompt — warmup({warmup}) + measure({iterations})")

        # Warmup
        for _ in range(warmup):
            benchmark_single_inference(model, primary_image, text, bbox)

        # Full pipeline (cold per-image)
        result = BenchmarkResult(
            variant=label,
            config_name=config_name,
            prompt_type=prompt_label,
        )
        for _ in range(iterations):
            timing = benchmark_single_inference(model, primary_image, text, bbox)
            result.timings.append(timing)
        results.append(result)

        # Cached vision features (decoder-only)
        cached_result = BenchmarkResult(
            variant=label,
            config_name=config_name,
            prompt_type=f"{prompt_label}(cached-vis)",
        )
        cached_timings = benchmark_cached_vision(
            model,
            primary_image,
            text,
            bbox,
            iterations=iterations,
        )
        cached_result.timings = cached_timings
        results.append(cached_result)

    console.print(f"  [green]live-stream[/green] — warmup({warmup}) + measure({live_frames})")

    # Warmup
    for _ in range(warmup):
        benchmark_single_inference(model, real_images[0], "elephant", None)

    live_result = BenchmarkResult(
        variant=label,
        config_name=config_name,
        prompt_type="live-stream",
    )
    live_timings = benchmark_live_stream(
        model,
        real_images,
        "elephant",
        iterations=live_frames,
    )
    live_result.timings = live_timings
    results.append(live_result)

    return results


def run_benchmarks(
    base_dir: Path | None,
    variants: list[str],
    device: str = "CPU",
    warmup: int = 3,
    iterations: int = 10,
    live_frames: int = 15,
    ov_configs: dict[str, dict] | None = None,
) -> list[BenchmarkResult]:
    """Run all benchmarks across variants, prompt types, and configs."""
    # Prepare images
    coco_dir = Path("examples/assets/coco")
    real_images: list[torch.Tensor] = []
    for img_file in sorted(coco_dir.glob("*.jpg"))[:4]:
        img = _make_real_image(img_file)
        if img is not None:
            real_images.append(img)

    if not real_images:
        console.print("[yellow]No real images found, using dummy images[/yellow]")
        real_images = [_make_dummy_image() for _ in range(3)]

    primary_image = real_images[0]

    # A box prompt (bounding box in xyxy for an elephant in 000000286874.jpg)
    sample_bbox = np.array([[216, 184, 458, 436]])

    # Point-as-box (expand a point to a small 20x20 box)
    point_center = np.array([337, 310])  # center of elephant
    half = 10
    point_bbox = np.array([
        [
            point_center[0] - half,
            point_center[1] - half,
            point_center[0] + half,
            point_center[1] + half,
        ],
    ])

    # Prompt configurations: (label, text, bbox_or_none)
    prompt_configs: list[tuple[str, str, np.ndarray | None]] = [
        ("text", "elephant", None),
        ("box", "visual", sample_bbox),
        ("point-as-box", "visual", point_bbox),
    ]

    all_results: list[BenchmarkResult] = []

    for variant_dir_name in variants:
        # Resolve variant path: local directory or auto-download from HuggingFace
        variant_path = _resolve_variant_path(base_dir, variant_dir_name)
        if variant_path is None:
            continue

        label = DEFAULT_VARIANTS.get(variant_dir_name, variant_dir_name)

        if ov_configs is None:
            ov_configs = _get_ov_configs(device)
        for config_name, config in ov_configs.items():
            console.rule(f"[bold cyan]{label} — {config_name}[/bold cyan]")

            try:
                model = BenchmarkModel(variant_path, device, config)
            except Exception:
                logger.exception("Failed to load %s with config %s", variant_dir_name, config_name)
                continue

            try:
                variant_results = _benchmark_variant_config(
                    model=model,
                    label=label,
                    config_name=config_name,
                    prompt_configs=prompt_configs,
                    primary_image=primary_image,
                    real_images=real_images,
                    warmup=warmup,
                    iterations=iterations,
                    live_frames=live_frames,
                )
                all_results.extend(variant_results)
            except RuntimeError as exc:
                console.print(f"[red]Runtime error for {label}/{config_name}: {exc}[/red]")
                console.print("[yellow]Skipping — likely GPU out-of-memory[/yellow]")
                continue

            # Cleanup
            del model
            gc.collect()

    return all_results



def print_summary_table(results: list[BenchmarkResult]) -> None:
    """Print a rich summary table of all benchmark results."""
    table = Table(
        title="SAM3 OpenVINO Benchmark Results (ms)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Variant", style="cyan", width=16)
    table.add_column("Config", width=16)
    table.add_column("Prompt", width=18)
    table.add_column("Preproc", justify="right")
    table.add_column("Vision Enc", justify="right")
    table.add_column("Text Enc", justify="right")
    table.add_column("Geo Enc", justify="right")
    table.add_column("Decoder", justify="right")
    table.add_column("Postproc", justify="right")
    table.add_column("Total", justify="right", style="bold")
    table.add_column("w/o Vision", justify="right", style="green")
    table.add_column("FPS", justify="right", style="magenta")

    for r in results:
        s_pre = r.stats("preprocess_ms")
        s_vis = r.stats("vision_encoder_ms")
        s_txt = r.stats("text_encoder_ms")
        s_geo = r.stats("geometry_encoder_ms")
        s_dec = r.stats("decoder_ms")
        s_post = r.stats("postprocess_ms")
        s_total = r.stats("total_ms")
        s_novis = r.stats("without_vision_ms")

        total_mean = s_total["mean"]
        fps = 1000.0 / total_mean if total_mean > 0 else 0

        table.add_row(
            r.variant,
            r.config_name,
            r.prompt_type,
            f"{s_pre['mean']:.1f}",
            f"{s_vis['mean']:.1f}",
            f"{s_txt['mean']:.1f}",
            f"{s_geo['mean']:.1f}",
            f"{s_dec['mean']:.1f}",
            f"{s_post['mean']:.1f}",
            f"{s_total['mean']:.1f}",
            f"{s_novis['mean']:.1f}",
            f"{fps:.2f}",
        )

    console.print(table)


def print_component_detail(results: list[BenchmarkResult]) -> None:
    """Print detailed component statistics for each benchmark."""
    for r in results:
        console.print(
            f"\n[bold]{r.variant} | {r.config_name} | {r.prompt_type}[/bold]",
        )
        detail_table = Table(show_header=True, box=None, pad_edge=False)
        detail_table.add_column("Component", width=18)
        detail_table.add_column("Mean", justify="right")
        detail_table.add_column("Median", justify="right")
        detail_table.add_column("Std", justify="right")
        detail_table.add_column("Min", justify="right")
        detail_table.add_column("Max", justify="right")

        for attr_name, label in [
            ("preprocess_ms", "Preprocess"),
            ("vision_encoder_ms", "Vision Encoder"),
            ("text_encoder_ms", "Text Encoder"),
            ("geometry_encoder_ms", "Geometry Encoder"),
            ("decoder_ms", "Prompt Decoder"),
            ("postprocess_ms", "Postprocess"),
            ("total_ms", "TOTAL"),
            ("without_vision_ms", "w/o Vision"),
        ]:
            s = r.stats(attr_name)
            style = "bold" if attr_name == "total_ms" else ""
            detail_table.add_row(
                label,
                f"{s['mean']:.1f}",
                f"{s['median']:.1f}",
                f"{s['std']:.1f}",
                f"{s['min']:.1f}",
                f"{s['max']:.1f}",
                style=style,
            )
        console.print(detail_table)


def print_comparison_matrix(results: list[BenchmarkResult]) -> None:
    """Print a compact variant x prompt comparison (default config only)."""
    # Filter to default config, full pipeline only (not cached / live-stream)
    default_results = [
        r
        for r in results
        if r.config_name == "default" and "cached" not in r.prompt_type and r.prompt_type != "live-stream"
    ]

    if not default_results:
        return

    table = Table(title="Variant x Prompt Comparison (default config, ms)", show_header=True)
    table.add_column("Variant", style="cyan", width=16)

    prompt_types = sorted({r.prompt_type for r in default_results})
    for pt in prompt_types:
        table.add_column(f"{pt} (total)", justify="right")
        table.add_column(f"{pt} (geo+dec)", justify="right")

    variant_names = sorted({r.variant for r in default_results})
    for variant in variant_names:
        row = [variant]
        for pt in prompt_types:
            match = [r for r in default_results if r.variant == variant and r.prompt_type == pt]
            if match:
                s_total = match[0].stats("total_ms")
                s_geo = match[0].stats("geometry_encoder_ms")
                s_dec = match[0].stats("decoder_ms")
                geo_dec = s_geo["mean"] + s_dec["mean"]
                row.extend([f"{s_total['mean']:.1f}", f"{geo_dec:.1f}"])
            else:
                row.extend(["—", "—"])
        table.add_row(*row)

    console.print(table)


def print_speedup_summary(results: list[BenchmarkResult]) -> None:
    """Print speedup of quantized variants relative to FP16 (default config)."""
    default_full = [
        r
        for r in results
        if r.config_name == "default"
        and "cached" not in r.prompt_type
        and r.prompt_type != "live-stream"
        and r.prompt_type == "text"
    ]

    fp16 = [r for r in default_full if "FP16" in r.variant and "ONNX" not in r.variant]
    if not fp16:
        return

    baseline = fp16[0].stats("total_ms")["mean"]
    baseline_vis = fp16[0].stats("vision_encoder_ms")["mean"]

    table = Table(title="Speedup vs OV-FP16 (text prompt, default config)", show_header=True)
    table.add_column("Variant", style="cyan", width=16)
    table.add_column("Total (ms)", justify="right")
    table.add_column("Speedup", justify="right", style="green")
    table.add_column("Vision Enc (ms)", justify="right")
    table.add_column("Vis Speedup", justify="right", style="green")

    for r in default_full:
        total = r.stats("total_ms")["mean"]
        vis = r.stats("vision_encoder_ms")["mean"]
        total_speedup = baseline / total if total > 0 else 0
        vis_speedup = baseline_vis / vis if vis > 0 else 0

        table.add_row(
            r.variant,
            f"{total:.1f}",
            f"{total_speedup:.2f}x",
            f"{vis:.1f}",
            f"{vis_speedup:.2f}x",
        )

    console.print(table)


def print_live_fps_table(results: list[BenchmarkResult]) -> None:
    """Print live-stream FPS comparison across variants and configs."""
    live_results = [r for r in results if r.prompt_type == "live-stream"]
    if not live_results:
        return

    table = Table(title="Live Stream FPS (text prompt, cached text encoder)", show_header=True)
    table.add_column("Variant", style="cyan", width=16)
    table.add_column("Config", width=16)
    table.add_column("Total (ms)", justify="right")
    table.add_column("Vision (ms)", justify="right")
    table.add_column("Decoder (ms)", justify="right")
    table.add_column("FPS", justify="right", style="bold magenta")

    for r in live_results:
        s = r.stats("total_ms")
        s_vis = r.stats("vision_encoder_ms")
        s_dec = r.stats("decoder_ms")
        fps = 1000.0 / s["mean"] if s["mean"] > 0 else 0

        table.add_row(
            r.variant,
            r.config_name,
            f"{s['mean']:.1f}",
            f"{s_vis['mean']:.1f}",
            f"{s_dec['mean']:.1f}",
            f"{fps:.1f}",
        )

    console.print(table)



def _get_device_full_name(device: str) -> str:
    """Get the full device name as reported by OpenVINO.

    Args:
        device: OpenVINO device string (e.g. ``CPU``, ``GPU``).

    Returns:
        Human-readable device name (e.g. ``Intel Core Ultra 7 265K``).
    """
    core = ov.Core()
    try:
        return core.get_property(device, "FULL_DEVICE_NAME")
    except RuntimeError:
        return device


def save_charts(
    results: list[BenchmarkResult],
    device: str,
    output_dir: Path,
) -> Path | None:
    """Generate benchmark comparison charts and save as PNG.

    Creates a multi-panel figure with:
      - Bar chart: mean end-to-end latency per variant and prompt type
      - Bar chart: component breakdown (stacked) per variant
      - Bar chart: live-stream FPS per variant

    Args:
        results: Benchmark results to visualize.
        device: OpenVINO device string used for benchmarking.
        output_dir: Directory to write the chart PNG.

    Returns:
        Path to the saved chart, or ``None`` if matplotlib is unavailable.
    """
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        console.print("[yellow]matplotlib not installed — skipping chart generation[/yellow]")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = _get_device_full_name(device)

    # Filter to default config for cleaner charts
    default_full = [
        r
        for r in results
        if r.config_name == "default" and "cached" not in r.prompt_type and r.prompt_type != "live-stream"
    ]
    live_results = [r for r in results if r.prompt_type == "live-stream" and r.config_name == "default"]

    if not default_full:
        console.print("[yellow]No default-config results for charting[/yellow]")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"SAM3 OpenVINO Benchmark — {device_name}", fontsize=14, fontweight="bold")

    # --- Panel 1: End-to-end latency by variant × prompt type ---
    ax1 = axes[0]
    variants = sorted({r.variant for r in default_full})
    prompt_types = sorted({r.prompt_type for r in default_full})
    bar_width = 0.8 / max(len(prompt_types), 1)

    x = np.arange(len(variants))
    for i, pt in enumerate(prompt_types):
        means = []
        for v in variants:
            match = [r for r in default_full if r.variant == v and r.prompt_type == pt]
            means.append(match[0].stats("total_ms")["mean"] if match else 0)
        ax1.bar(x + i * bar_width, means, bar_width, label=pt)

    ax1.set_xlabel("Model Variant")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("End-to-End Latency")
    ax1.set_xticks(x + bar_width * (len(prompt_types) - 1) / 2)
    ax1.set_xticklabels(variants, rotation=15, ha="right", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # --- Panel 2: Component breakdown (stacked bar, text prompt) ---
    ax2 = axes[1]
    text_results = [r for r in default_full if r.prompt_type == "text"]
    components = [
        ("preprocess_ms", "Preprocess"),
        ("vision_encoder_ms", "Vision Encoder"),
        ("text_encoder_ms", "Text Encoder"),
        ("geometry_encoder_ms", "Geometry Encoder"),
        ("decoder_ms", "Decoder"),
        ("postprocess_ms", "Postprocess"),
    ]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#64b5cd"]

    text_variants = [r.variant for r in text_results]
    x2 = np.arange(len(text_variants))
    bottom = np.zeros(len(text_variants))

    for (attr, label), color in zip(components, colors, strict=False):
        vals = [r.stats(attr)["mean"] for r in text_results]
        ax2.bar(x2, vals, bottom=bottom, label=label, color=color, width=0.6)
        bottom += np.array(vals)

    ax2.set_xlabel("Model Variant")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Component Breakdown (text prompt)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(text_variants, rotation=15, ha="right", fontsize=9)
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(axis="y", alpha=0.3)

    # --- Panel 3: Live-stream FPS ---
    ax3 = axes[2]
    if live_results:
        live_variants = [r.variant for r in live_results]
        fps_values = [1000.0 / r.stats("total_ms")["mean"] if r.stats("total_ms")["mean"] > 0 else 0 for r in live_results]
        x3 = np.arange(len(live_variants))
        bars = ax3.bar(x3, fps_values, color="#55a868", width=0.6)

        for bar, fps in zip(bars, fps_values, strict=False):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{fps:.1f}", ha="center", va="bottom", fontsize=9)

        ax3.set_xlabel("Model Variant")
        ax3.set_ylabel("FPS")
        ax3.set_title("Live Stream FPS (text prompt)")
        ax3.set_xticks(x3)
        ax3.set_xticklabels(live_variants, rotation=15, ha="right", fontsize=9)
        ax3.grid(axis="y", alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No live-stream data", transform=ax3.transAxes, ha="center", va="center")
        ax3.set_title("Live Stream FPS")

    plt.tight_layout()

    device_tag = device.lower().replace(".", "_")
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    chart_path = output_dir / f"sam3_benchmark_{device_tag}_{timestamp}.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    console.print(f"[bold green]Chart saved to:[/bold green] {chart_path}")
    return chart_path


def _results_to_dataframe(results: list[BenchmarkResult]):  # noqa: ANN202
    """Convert benchmark results to a pandas DataFrame with per-row statistics.

    Each row represents one (variant, config, prompt_type) combination with
    mean, median, std, min, max for every timing component.

    Requires ``pandas`` (install with ``uv pip install pandas``).

    Raises:
        ImportError: If ``pandas`` is not installed.
    """
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError as exc:
        msg = "pandas is required to export results. Install it with: uv pip install pandas openpyxl"
        raise ImportError(msg) from exc

    rows: list[dict[str, object]] = []

    components = [
        ("preprocess_ms", "Preprocess"),
        ("vision_encoder_ms", "Vision Encoder"),
        ("text_encoder_ms", "Text Encoder"),
        ("geometry_encoder_ms", "Geometry Encoder"),
        ("decoder_ms", "Prompt Decoder"),
        ("postprocess_ms", "Postprocess"),
        ("total_ms", "Total"),
        ("without_vision_ms", "w/o Vision"),
    ]

    for r in results:
        row: dict[str, object] = {
            "Variant": r.variant,
            "Config": r.config_name,
            "Prompt": r.prompt_type,
            "Iterations": len(r.timings),
        }
        total_mean = r.stats("total_ms")["mean"]
        row["FPS"] = round(1000.0 / total_mean, 2) if total_mean > 0 else 0

        for attr, label in components:
            s = r.stats(attr)
            row[f"{label} Mean (ms)"] = round(s["mean"], 2)
            row[f"{label} Median (ms)"] = round(s["median"], 2)
            row[f"{label} Std (ms)"] = round(s["std"], 2)
            row[f"{label} Min (ms)"] = round(s["min"], 2)
            row[f"{label} Max (ms)"] = round(s["max"], 2)

        rows.append(row)

    return pd.DataFrame(rows)


def save_results(
    results: list[BenchmarkResult],
    device: str,
    output_dir: Path,
) -> Path:
    """Save benchmark results to an Excel file.

    The file is named ``sam3_benchmark_{device}_{timestamp}.xlsx`` and includes
    a metadata row with the device name as reported by OpenVINO.

    Args:
        results: Benchmark results to save.
        device: OpenVINO device string used for benchmarking.
        output_dir: Directory to write the output file.

    Returns:
        Path to the saved file.

    Raises:
        ImportError: If ``pandas`` or ``openpyxl`` is not installed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = _get_device_full_name(device)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    device_tag = device.lower().replace(".", "_")
    filename = f"sam3_benchmark_{device_tag}_{timestamp}.xlsx"
    filepath = output_dir / filename

    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError as exc:
        msg = "pandas and openpyxl are required to save results. Install with: uv pip install pandas openpyxl"
        raise ImportError(msg) from exc

    dataframe = _results_to_dataframe(results)

    # Metadata DataFrame
    meta_df = pd.DataFrame([
        {"Key": "Device", "Value": device},
        {"Key": "Device Full Name", "Value": device_name},
        {"Key": "Timestamp (UTC)", "Value": timestamp},
        {"Key": "OpenVINO Version", "Value": ov.get_version()},
    ])

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        meta_df.to_excel(writer, sheet_name="Device Info", index=False)
        dataframe.to_excel(writer, sheet_name="Benchmark Results", index=False)

    console.print(f"\n[bold green]Results saved to:[/bold green] {filepath}")
    console.print(f"  Device: [cyan]{device_name}[/cyan]")
    return filepath



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark SAM3 OpenVINO inference latency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help=(
            "Local directory with model variant subdirectories."
            " If omitted, models are auto-downloaded from HuggingFace."
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS.keys()),
        help="Model variant directory names to benchmark.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help='OpenVINO device: "CPU", "GPU" (Intel iGPU/dGPU/XPU), or "AUTO". Default: CPU',
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations before measurement.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of measured iterations per benchmark.",
    )
    parser.add_argument(
        "--live-frames",
        type=int,
        default=15,
        help="Number of frames for live-stream benchmark.",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Print detailed per-component statistics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./benchmark_results"),
        help="Directory to save benchmark results Excel file. Default: ./benchmark_results",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation (requires matplotlib).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "OpenVINO model cache directory. Caches compiled GPU kernels so "
            "JIT compilation only runs once. Strongly recommended for GPU to "
            "avoid driver crashes. Default: auto-set to ./ov_cache for GPU."
        ),
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

    console.rule("[bold]SAM3 OpenVINO Benchmark[/bold]")
    device_name = _get_device_full_name(args.device)
    console.print(f"Device: {args.device} ([cyan]{device_name}[/cyan])")
    if _is_gpu_available():
        gpu_name = _get_device_full_name("GPU")
        console.print(f"Intel GPU available: [cyan]{gpu_name}[/cyan]")
    console.print(f"Warmup: {args.warmup}, Iterations: {args.iterations}, Live frames: {args.live_frames}")
    console.print(f"Variants: {args.variants}")
    if args.base_dir is not None:
        console.print(f"Model source: local ({args.base_dir})")
    else:
        console.print(f"Model source: HuggingFace ({SAM3_HF_REPO_ID})")

    # Default to ./ov_cache for GPU if no cache dir specified
    cache_dir: Path | None = args.cache_dir
    if cache_dir is None and args.device.upper().startswith("GPU"):
        cache_dir = Path("./ov_cache")
        console.print(f"[yellow]GPU detected — enabling model cache: {cache_dir}[/yellow]")

    ov_configs = _get_ov_configs(args.device, cache_dir=cache_dir)
    console.print(f"OV configs: {list(ov_configs.keys())}")
    console.print()

    results = run_benchmarks(
        base_dir=args.base_dir,
        variants=args.variants,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        live_frames=args.live_frames,
        ov_configs=ov_configs,
    )

    if not results:
        console.print("[red]No benchmark results collected.[/red]")
        return

    console.rule("[bold]Summary[/bold]")
    print_summary_table(results)

    console.rule("[bold]Variant x Prompt Comparison[/bold]")
    print_comparison_matrix(results)

    console.rule("[bold]Speedup vs FP16[/bold]")
    print_speedup_summary(results)

    console.rule("[bold]Live Stream FPS[/bold]")
    print_live_fps_table(results)

    if args.detail:
        console.rule("[bold]Detailed Statistics[/bold]")
        print_component_detail(results)

    # Save results to Excel
    save_results(results, args.device, args.output_dir)

    # Generate comparison charts
    if not args.no_charts:
        save_charts(results, args.device, args.output_dir)


if __name__ == "__main__":
    main()
