# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Full post-training quantization (PTQ) for SAM3 OpenVINO IR models.

Applies ``nncf.quantize()`` to each SAM3 sub-model using real calibration
data.  Unlike weight-only compression (``export_sam3.py --quantize``), this
quantizes **both weights and activations** to INT8, enabling hardware INT8
acceleration on CPUs with VNNI/AMX and GPUs with DP4A/DPAS.

Calibration data is built per sub-model:

* **vision-encoder** - real images resized and normalised to [-1, 1].
* **text-encoder** - LVIS-92 category names tokenised with CLIP tokenizer.
* **geometry-encoder / geometry-encoder-exemplar** - vision-encoder outputs
  paired with synthetic box/point prompts.
* **prompt-decoder** - chained vision-encoder + text-encoder outputs.

Usage:
    # Quantize all 5 sub-models (default: MIXED preset)
    python ptq_sam3.py \\
        --source-dir ./sam3-openvino/openvino-fp16 \\
        --calibration-dir ./calibration-images \\
        --output-dir ./sam3-openvino

    # Quantize only the vision encoder with GPU target
    python ptq_sam3.py \\
        --source-dir ./sam3-openvino/openvino-fp16 \\
        --calibration-dir ./calibration-images \\
        --output-dir ./sam3-openvino \\
        --models vision-encoder \\
        --target-device GPU

    # Quantize with Transformer model type hint and validate
    python ptq_sam3.py \\
        --source-dir ./sam3-openvino/openvino-fp16 \\
        --calibration-dir ./calibration-images \\
        --output-dir ./sam3-openvino \\
        --model-type Transformer \\
        --validate

See Also:
    :mod:`instantlearn.utils.compression` - ``quantize_model()`` wrapper
    :mod:`instantlearn.scripts.sam3.export_sam3` - ONNX export and weight-only compression
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Tokenizer files to copy alongside quantized models.
_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]


# Calibration data helpers

def collect_image_paths(calibration_dir: Path, max_images: int = 300) -> list[Path]:
    """Collect image file paths from a calibration directory.

    Args:
        calibration_dir: Directory containing calibration images.
        max_images: Maximum number of images to collect.

    Returns:
        List of image file paths sorted by name.

    Raises:
        FileNotFoundError: If no images are found in *calibration_dir*.
    """
    paths = sorted(
        p for p in calibration_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not paths:
        msg = f"No images found in {calibration_dir}"
        raise FileNotFoundError(msg)
    if len(paths) > max_images:
        paths = paths[:max_images]
    logger.info("Collected %d calibration images from %s", len(paths), calibration_dir)
    return paths


def preprocess_image(image_path: Path, target_size: int = 1008) -> np.ndarray:
    """Load and preprocess an image for the vision encoder.

    Resizes to ``(target_size, target_size)`` and normalises to [-1, 1].

    Args:
        image_path: Path to the image file.
        target_size: Target spatial resolution.

    Returns:
        Preprocessed image as float32 array with shape ``(1, 3, H, W)``.
    """
    from PIL import Image  # noqa: PLC0415

    img = Image.open(image_path).convert("RGB").resize((target_size, target_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    # Normalise to [-1, 1]
    arr = arr * 2.0 - 1.0
    # HWC → CHW, add batch dim
    return arr.transpose(2, 0, 1)[np.newaxis]


def _build_vision_encoder_calibration(
    image_paths: list[Path],
    target_size: int = 1008,
) -> list[dict[str, np.ndarray]]:
    """Build calibration dataset for the vision encoder.

    Args:
        image_paths: Calibration image file paths.
        target_size: Input resolution.

    Returns:
        List of dicts mapping ``"pixel_values"`` to preprocessed images.
    """
    samples = []
    for path in image_paths:
        pixel_values = preprocess_image(path, target_size)
        samples.append({"pixel_values": pixel_values})
    logger.info("Built %d vision-encoder calibration samples", len(samples))
    return samples


def _build_text_encoder_calibration(
    source_dir: Path,
    num_samples: int = 300,
) -> list[dict[str, np.ndarray]]:
    """Build calibration dataset for the text encoder.

    Uses LVIS-92 category names tokenised with the CLIP tokenizer stored
    alongside the OpenVINO IR models.

    Args:
        source_dir: Directory containing the FP16 OpenVINO IR + tokenizer files.
        num_samples: Number of calibration samples to generate.

    Returns:
        List of dicts with ``"input_ids"`` and ``"attention_mask"``.
    """
    from transformers import CLIPTokenizerFast  # noqa: PLC0415

    from instantlearn.utils.constants import LVIS_92_BENCHMARK_CATEGORIES  # noqa: PLC0415

    tokenizer = CLIPTokenizerFast.from_pretrained(str(source_dir))
    categories = LVIS_92_BENCHMARK_CATEGORIES["fold_0"]
    rng = np.random.default_rng(42)

    samples = []
    for _ in range(num_samples):
        # Sample 1-4 categories to form a text prompt
        n_cats = rng.integers(1, min(5, len(categories) + 1))
        chosen = rng.choice(categories, size=n_cats, replace=False).tolist()
        text = " . ".join(chosen)
        encoded = tokenizer(text, padding="max_length", max_length=32, truncation=True, return_tensors="np")
        samples.append({
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        })
    logger.info("Built %d text-encoder calibration samples", len(samples))
    return samples


def _build_geometry_encoder_calibration(
    vision_compiled: object,
    image_paths: list[Path],
    target_size: int = 1008,
    *,
    exemplar: bool = False,
) -> list[dict[str, np.ndarray]]:
    """Build calibration dataset for the geometry encoder.

    Runs the vision encoder on real images and pairs the FPN outputs with
    synthetic box/point prompts.

    Args:
        vision_compiled: Compiled OpenVINO vision-encoder model.
        image_paths: Calibration image file paths.
        target_size: Input resolution.
        exemplar: If True, build data for the exemplar variant (points only,
            boxes set to ignore label -10).

    Returns:
        List of calibration dicts matching geometry-encoder inputs.
    """
    rng = np.random.default_rng(42)
    samples = []

    for path in image_paths:
        pixel_values = preprocess_image(path, target_size)
        vision_out = vision_compiled([pixel_values])

        fpn_feat_2 = vision_out["fpn_feat_2"]
        fpn_pos_2 = vision_out["fpn_pos_2"]

        if exemplar:
            # Exemplar variant: points only, ignore boxes
            boxes = np.zeros((1, 1, 4), dtype=np.float32)
            box_labels = np.full((1, 1), -10, dtype=np.int64)
            points = rng.random((1, 1, 2)).astype(np.float32)
            point_labels = np.ones((1, 1), dtype=np.int64)
        else:
            # Classic variant: box + optional point
            boxes = rng.random((1, 1, 4)).astype(np.float32)
            box_labels = np.ones((1, 1), dtype=np.int64)
            points = np.zeros((1, 1, 2), dtype=np.float32)
            point_labels = np.full((1, 1), -10, dtype=np.int64)

        samples.append({
            "fpn_feat_2": fpn_feat_2,
            "fpn_pos_2": fpn_pos_2,
            "input_boxes": boxes,
            "input_boxes_labels": box_labels,
            "input_points": points,
            "input_points_labels": point_labels,
        })

    variant = "exemplar" if exemplar else "classic"
    logger.info("Built %d geometry-encoder (%s) calibration samples", len(samples), variant)
    return samples


def _build_prompt_decoder_calibration(
    vision_compiled: object,
    text_compiled: object,
    source_dir: Path,
    image_paths: list[Path],
    target_size: int = 1008,
) -> list[dict[str, np.ndarray]]:
    """Build calibration dataset for the prompt decoder.

    Chains the vision encoder and text encoder to produce realistic FPN
    features and prompt embeddings.

    Args:
        vision_compiled: Compiled OpenVINO vision-encoder model.
        text_compiled: Compiled OpenVINO text-encoder model.
        source_dir: Directory with tokenizer files.
        image_paths: Calibration image file paths.
        target_size: Input resolution.

    Returns:
        List of calibration dicts matching prompt-decoder inputs.
    """
    from transformers import CLIPTokenizerFast  # noqa: PLC0415

    from instantlearn.utils.constants import LVIS_92_BENCHMARK_CATEGORIES  # noqa: PLC0415

    tokenizer = CLIPTokenizerFast.from_pretrained(str(source_dir))
    categories = LVIS_92_BENCHMARK_CATEGORIES["fold_0"]
    rng = np.random.default_rng(42)
    samples = []

    for path in image_paths:
        pixel_values = preprocess_image(path, target_size)
        vision_out = vision_compiled([pixel_values])

        # Tokenize a random set of categories
        n_cats = rng.integers(1, min(5, len(categories) + 1))
        chosen = rng.choice(categories, size=n_cats, replace=False).tolist()
        text = " . ".join(chosen)
        encoded = tokenizer(text, padding="max_length", max_length=32, truncation=True, return_tensors="np")
        text_out = text_compiled([
            encoded["input_ids"].astype(np.int64),
            encoded["attention_mask"].astype(np.int64),
        ])

        samples.append({
            "fpn_feat_0": vision_out["fpn_feat_0"],
            "fpn_feat_1": vision_out["fpn_feat_1"],
            "fpn_feat_2": vision_out["fpn_feat_2"],
            "fpn_pos_2": vision_out["fpn_pos_2"],
            "prompt_features": text_out["text_features"],
            "prompt_mask": text_out["text_mask"],
        })

    logger.info("Built %d prompt-decoder calibration samples", len(samples))
    return samples


# Main quantization pipeline

def quantize_sam3_ptq(  # noqa: C901, PLR0915
    source_dir: Path,
    output_dir: Path,
    calibration_dir: Path,
    *,
    models: list[str] | None = None,
    target_device: str = "ANY",
    preset: str = "MIXED",
    model_type: str | None = None,
    num_calibration: int = 300,
    variant_name: str = "ptq-int8",
    target_size: int = 1008,
    validate: bool = False,
) -> Path:
    """Apply full INT8 PTQ to SAM3 OpenVINO IR models.

    Args:
        source_dir: Directory containing FP16 OpenVINO IR models.
        output_dir: Base output directory.
        calibration_dir: Directory containing calibration images.
        models: Sub-models to quantize. Defaults to all 5.
        target_device: NNCF target device (ANY, CPU, CPU_SPR, GPU, NPU).
        preset: Quantization preset (PERFORMANCE or MIXED).
        model_type: Optional model type hint (e.g. "Transformer").
        num_calibration: Number of calibration samples per sub-model.
        variant_name: Name for the output subdirectory.
        target_size: Input image resolution.
        validate: Whether to validate quantized models after export.

    Returns:
        Path to directory containing quantized models.

    Raises:
        FileNotFoundError: If a required source model for calibration is missing.
    """
    import openvino as ov  # noqa: PLC0415

    from instantlearn.scripts.sam3.export_sam3 import MODEL_NAMES, validate_openvino_models  # noqa: PLC0415
    from instantlearn.utils.compression import quantize_model  # noqa: PLC0415

    if models is None:
        models = list(MODEL_NAMES)

    ir_dir = output_dir / f"openvino-{variant_name}"
    ir_dir.mkdir(parents=True, exist_ok=True)

    core = ov.Core()
    image_paths = collect_image_paths(calibration_dir, max_images=num_calibration)

    # Pre-compile shared models for calibration data generation
    vision_compiled = None
    text_compiled = None

    needs_vision = any(
        m in models
        for m in ["geometry-encoder", "geometry-encoder-exemplar", "prompt-decoder"]
    )
    needs_text = "prompt-decoder" in models

    if needs_vision:
        vision_xml = source_dir / "vision-encoder.xml"
        if not vision_xml.exists():
            msg = f"vision-encoder.xml required for calibration but not found in {source_dir}"
            raise FileNotFoundError(msg)
        vision_compiled = core.compile_model(vision_xml, "CPU")
        logger.info("Compiled vision-encoder for calibration data generation")

    if needs_text:
        text_xml = source_dir / "text-encoder.xml"
        if not text_xml.exists():
            msg = f"text-encoder.xml required for calibration but not found in {source_dir}"
            raise FileNotFoundError(msg)
        text_compiled = core.compile_model(text_xml, "CPU")
        logger.info("Compiled text-encoder for calibration data generation")

    # Quantize each sub-model
    for model_name in models:
        xml_path = source_dir / f"{model_name}.xml"
        if not xml_path.exists():
            logger.warning("Skipping %s — not found in %s", model_name, source_dir)
            continue

        logger.info("Building calibration data for %s...", model_name)
        if model_name == "vision-encoder":
            cal_data = _build_vision_encoder_calibration(image_paths, target_size)
        elif model_name == "text-encoder":
            cal_data = _build_text_encoder_calibration(source_dir, num_calibration)
        elif model_name == "geometry-encoder":
            cal_data = _build_geometry_encoder_calibration(
                vision_compiled, image_paths, target_size, exemplar=False,
            )
        elif model_name == "geometry-encoder-exemplar":
            cal_data = _build_geometry_encoder_calibration(
                vision_compiled, image_paths, target_size, exemplar=True,
            )
        elif model_name == "prompt-decoder":
            cal_data = _build_prompt_decoder_calibration(
                vision_compiled, text_compiled, source_dir, image_paths, target_size,
            )
        else:
            logger.warning("Unknown model: %s", model_name)
            continue

        logger.info("Quantizing %s (device=%s, preset=%s)...", model_name, target_device, preset)
        ov_model = core.read_model(xml_path)
        quantized = quantize_model(
            ov_model,
            cal_data,
            target_device=target_device,
            preset=preset,
            subset_size=min(num_calibration, len(cal_data)),
            model_type=model_type,
        )

        out_xml = ir_dir / f"{model_name}.xml"
        ov.save_model(quantized, str(out_xml))
        bin_path = ir_dir / f"{model_name}.bin"
        size_mb = bin_path.stat().st_size / (1024 * 1024)
        logger.info("Saved: %s (%.1f MB)", out_xml, size_mb)

    # Copy tokenizer files
    for filename in _TOKENIZER_FILES:
        src = source_dir / filename
        dst = ir_dir / filename
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    # Copy any non-quantized models (so the output dir is self-contained)
    for model_name in MODEL_NAMES:
        if model_name not in models:
            for ext in (".xml", ".bin"):
                src = source_dir / f"{model_name}{ext}"
                dst = ir_dir / f"{model_name}{ext}"
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
                    logger.info("Copied non-quantized %s%s", model_name, ext)

    if validate:
        validate_openvino_models(ir_dir, resolution=target_size)

    logger.info("PTQ complete. Quantized models saved to %s", ir_dir)
    return ir_dir


# CLI

def main() -> None:
    """CLI entry point for SAM3 full post-training quantization."""
    from instantlearn.scripts.sam3.export_sam3 import MODEL_NAMES  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Apply full INT8 post-training quantization to SAM3 OpenVINO IR models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize all sub-models
  python ptq_sam3.py --source-dir ./sam3-openvino/openvino-fp16 \\
      --calibration-dir ./calibration-images --output-dir ./sam3-openvino

  # Quantize only vision encoder targeting GPU
  python ptq_sam3.py --source-dir ./sam3-openvino/openvino-fp16 \\
      --calibration-dir ./calibration-images --output-dir ./sam3-openvino \\
      --models vision-encoder --target-device GPU

  # Quantize with Transformer hint and validate
  python ptq_sam3.py --source-dir ./sam3-openvino/openvino-fp16 \\
      --calibration-dir ./calibration-images --output-dir ./sam3-openvino \\
      --model-type Transformer --validate
        """,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory with FP16 OpenVINO IR models to quantize.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        required=True,
        help="Directory containing calibration images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./sam3-openvino"),
        help="Base output directory. Default: ./sam3-openvino",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=MODEL_NAMES,
        default=None,
        help="Sub-models to quantize. Default: all 5.",
    )
    parser.add_argument(
        "--target-device",
        type=str,
        choices=["ANY", "CPU", "CPU_SPR", "GPU", "NPU"],
        default="ANY",
        help="NNCF target device. Default: ANY",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["PERFORMANCE", "MIXED"],
        default="MIXED",
        help="Quantization preset. Default: MIXED",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["Transformer"],
        default=None,
        help="Model type hint for NNCF. Use 'Transformer' for attention models.",
    )
    parser.add_argument(
        "--num-calibration",
        type=int,
        default=300,
        help="Number of calibration samples per sub-model. Default: 300",
    )
    parser.add_argument(
        "--variant-name",
        type=str,
        default="ptq-int8",
        help="Name for the output subdirectory. Default: ptq-int8",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1008,
        help="Input image resolution. Default: 1008",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate quantized models with dummy inference.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(name)s: %(message)s",
        stream=sys.stdout,
    )

    quantize_sam3_ptq(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        calibration_dir=args.calibration_dir,
        models=args.models,
        target_device=args.target_device,
        preset=args.preset,
        model_type=args.model_type,
        num_calibration=args.num_calibration,
        variant_name=args.variant_name,
        target_size=args.resolution,
        validate=args.validate,
    )


if __name__ == "__main__":
    main()
