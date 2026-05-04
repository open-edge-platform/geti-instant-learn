# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Quantize SAM3 models to INT8/INT4 for faster inference with OpenVINO.

Applies NNCF weight compression (INT8 or INT4) to existing FP16 OpenVINO IR
models.  Produces proper OpenVINO IR with compressed weights.  No calibration
data is needed.

Uses the shared ``compress_model()`` utility from
``instantlearn.utils.compression``.

Usage:
    # Apply INT8 symmetric weight compression to FP16 IR
    python quantize_sam3_openvino.py --compression int8_sym --source-dir ./sam3-openvino/openvino-fp16

    # Apply INT4 symmetric weight compression to FP16 IR
    python quantize_sam3_openvino.py --compression int4_sym --source-dir ./sam3-openvino/openvino-fp16

    # Run all compression modes and compare sizes
    python quantize_sam3_openvino.py --compression all --source-dir ./sam3-openvino/openvino-fp16

    # Validate quantized models
    python quantize_sam3_openvino.py --compression int8_sym --source-dir ./sam3-openvino/openvino-fp16 --validate
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import openvino as ov
from rich.console import Console
from rich.table import Table

from instantlearn.utils.compression import compress_model
from instantlearn.utils.constants import CompressionMode

logger = logging.getLogger(__name__)

# Canonical model names (5-model split)
MODEL_NAMES = [
    "vision-encoder",
    "text-encoder",
    "geometry-encoder",
    "geometry-encoder-exemplar",
    "prompt-decoder",
]

# Compression modes applicable to SAM3 (excludes FP32/FP16 which are not quantisation)
_SAM3_COMPRESSION_MODES = [
    CompressionMode.INT8_SYM,
    CompressionMode.INT8_ASYM,
    CompressionMode.INT4_SYM,
    CompressionMode.INT4_ASYM,
]

# Tokenizer files needed for inference
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]


def apply_weight_compression(
    source_dir: Path,
    output_dir: Path,
    mode: CompressionMode = CompressionMode.INT8_SYM,
) -> Path:
    """Apply weight compression to SAM3 OpenVINO IR models.

    Args:
        source_dir: Directory containing FP16 OpenVINO IR models.
        output_dir: Base output directory.
        mode: Compression mode from :class:`CompressionMode`.

    Returns:
        Path to directory containing compressed OpenVINO IR models.
    """
    logger.info("=" * 60)
    logger.info("Applying %s weight compression", mode.value.upper())
    logger.info("=" * 60)

    ir_dir = output_dir / f"openvino-{mode.value}"
    ir_dir.mkdir(parents=True, exist_ok=True)

    core = ov.Core()
    logger.info("Compressing %d models", len(MODEL_NAMES))

    for model_name in MODEL_NAMES:
        xml_path = source_dir / f"{model_name}.xml"
        if not xml_path.exists():
            logger.warning("Source model not found: %s", xml_path)
            continue

        logger.info("Compressing %s with %s...", model_name, mode.value.upper())

        ov_model = core.read_model(xml_path)

        try:
            compressed_model = compress_model(ov_model, mode=mode, group_size=-1)
        except Exception:
            logger.exception("Failed to compress %s with %s", model_name, mode.value)
            continue

        out_xml = ir_dir / f"{model_name}.xml"
        ov.save_model(compressed_model, out_xml)

        bin_path = ir_dir / f"{model_name}.bin"
        size_mb = bin_path.stat().st_size / (1024 * 1024)
        logger.info("Saved: %s (%.1f MB)", out_xml, size_mb)

    # Copy tokenizer files from source
    for filename in TOKENIZER_FILES:
        src = source_dir / filename
        dst = ir_dir / filename
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    return ir_dir


def validate_openvino_models(model_dir: Path, device: str = "CPU") -> bool:  # noqa: C901, PLR0915
    """Validate that OpenVINO models can be loaded and run with dummy inputs.

    Tries .xml first, then falls back to .onnx files.

    Args:
        model_dir: Directory containing model files.
        device: OpenVINO device for validation.

    Returns:
        True if all models validated successfully.
    """
    core = ov.Core()
    rng = np.random.default_rng(42)
    all_ok = True

    logger.info("Validating models in %s ...", model_dir)

    model_names = MODEL_NAMES

    # Find model files (prefer .xml, fallback to .onnx)
    model_files = {}
    for model_name in model_names:
        xml = model_dir / f"{model_name}.xml"
        if xml.exists():
            model_files[model_name] = xml
            continue
        # Look for any .onnx variant
        onnx_candidates = sorted(model_dir.glob(f"{model_name}*.onnx"))
        if onnx_candidates:
            model_files[model_name] = onnx_candidates[0]
        else:
            logger.warning("  %s: NOT FOUND", model_name)
            all_ok = False

    # Validate vision encoder
    if "vision-encoder" in model_files:
        try:
            model = core.compile_model(model_files["vision-encoder"], device)
            dummy = rng.standard_normal((1, 3, 1008, 1008)).astype(np.float32)
            result = model([dummy])
            shapes = {name: result[name].shape for name in ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"]}
            logger.info("  vision-encoder: OK — %s", shapes)
        except Exception:
            logger.exception("  vision-encoder: FAILED")
            all_ok = False

    # Validate text encoder
    if "text-encoder" in model_files:
        try:
            model = core.compile_model(model_files["text-encoder"], device)
            dummy_ids = np.ones((1, 32), dtype=np.int64)
            dummy_mask = np.ones((1, 32), dtype=np.int64)
            result = model([dummy_ids, dummy_mask])
            shapes = {name: result[name].shape for name in ["text_features", "text_mask"]}
            logger.info("  text-encoder: OK — %s", shapes)
        except Exception:
            logger.exception("  text-encoder: FAILED")
            all_ok = False

    # Geometry encoder (classic)
    if "geometry-encoder" in model_files:
        try:
            model = core.compile_model(model_files["geometry-encoder"], device)
            dummy_fpn2 = rng.standard_normal((1, 256, 72, 72)).astype(np.float32)
            dummy_pos2 = rng.standard_normal((1, 256, 72, 72)).astype(np.float32)
            result = model([
                dummy_fpn2,
                dummy_pos2,
                rng.random((1, 1, 4)).astype(np.float32),
                np.ones((1, 1), dtype=np.int64),
                np.zeros((1, 1, 2), dtype=np.float32),
                np.full((1, 1), -10, dtype=np.int64),
            ])
            shapes = {name: result[name].shape for name in ["geometry_features", "geometry_mask"]}
            logger.info("  geometry-encoder: OK — %s", shapes)
        except Exception:
            logger.exception("  geometry-encoder: FAILED")
            all_ok = False

    # Geometry encoder (exemplar)
    if "geometry-encoder-exemplar" in model_files:
        try:
            model = core.compile_model(model_files["geometry-encoder-exemplar"], device)
            dummy_fpn2 = rng.standard_normal((1, 256, 72, 72)).astype(np.float32)
            dummy_pos2 = rng.standard_normal((1, 256, 72, 72)).astype(np.float32)
            result = model([
                dummy_fpn2,
                dummy_pos2,
                np.zeros((1, 1, 4), dtype=np.float32),
                np.full((1, 1), -10, dtype=np.int64),
                rng.random((1, 1, 2)).astype(np.float32),
                np.ones((1, 1), dtype=np.int64),
            ])
            shapes = {name: result[name].shape for name in ["geometry_features", "geometry_mask"]}
            logger.info("  geometry-encoder-exemplar: OK — %s", shapes)
        except Exception:
            logger.exception("  geometry-encoder-exemplar: FAILED")
            all_ok = False

    # Prompt decoder
    if "prompt-decoder" in model_files:
        try:
            model = core.compile_model(model_files["prompt-decoder"], device)
            result = model([
                rng.standard_normal((1, 256, 288, 288)).astype(np.float32),
                rng.standard_normal((1, 256, 144, 144)).astype(np.float32),
                rng.standard_normal((1, 256, 72, 72)).astype(np.float32),
                rng.standard_normal((1, 256, 72, 72)).astype(np.float32),
                rng.standard_normal((1, 32, 256)).astype(np.float32),
                np.ones((1, 32), dtype=bool),
            ])
            shapes = {
                name: result[name].shape for name in ["pred_masks", "pred_boxes", "pred_logits", "presence_logits"]
            }
            logger.info("  prompt-decoder: OK — %s", shapes)
        except Exception:
            logger.exception("  prompt-decoder: FAILED")
            all_ok = False

    status = "All models validated!" if all_ok else "Some models failed validation."
    logger.info(status)
    return all_ok


def get_dir_size(directory: Path) -> float:
    """Get total size of model files in a directory in MB.

    Args:
        directory: Directory to measure.

    Returns:
        Total size in megabytes.
    """
    total = 0
    for ext in ("*.xml", "*.bin", "*.onnx"):
        for f in directory.glob(ext):
            total += f.stat().st_size
    return total / (1024 * 1024)


def print_comparison_table(output_dir: Path) -> None:
    """Print a comparison table of all quantized variants.

    Args:
        output_dir: Base output directory containing variant subdirectories.
    """
    console = Console()
    table = Table(title="SAM3 Quantization Comparison", show_header=True)
    table.add_column("Variant", style="cyan", width=20)
    table.add_column("Total", justify="right", style="bold")
    table.add_column("Model Count", justify="right")
    table.add_column("Status", style="green")

    # Find all variant directories (openvino-* and onnx-*)
    variant_dirs = sorted([*output_dir.glob("openvino-*"), *output_dir.glob("onnx-*")])
    for variant_dir in variant_dirs:
        if not variant_dir.is_dir():
            continue
        variant_name = variant_dir.name
        for prefix in ("openvino-", "onnx-"):
            variant_name = variant_name.replace(prefix, "", 1) if variant_name.startswith(prefix) else variant_name
        fmt = "IR" if variant_dir.name.startswith("openvino") else "ONNX"
        variant_label = f"{variant_name} ({fmt})"

        model_names = MODEL_NAMES
        total_size = 0.0
        found = 0
        for model_name in model_names:
            bin_path = variant_dir / f"{model_name}.bin"
            if bin_path.exists():
                total_size += bin_path.stat().st_size / (1024 * 1024)
                found += 1
            else:
                onnx_files = list(variant_dir.glob(f"{model_name}*.onnx"))
                if onnx_files:
                    total_size += onnx_files[0].stat().st_size / (1024 * 1024)
                    found += 1

        status = "OK" if found == len(model_names) else f"Missing {len(model_names) - found}"

        table.add_row(
            variant_label,
            f"{total_size:.1f} MB",
            f"{found}/{len(model_names)}",
            status,
        )

    console.print(table)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the quantization CLI.

    Returns:
        Configured argument parser.
    """
    valid_modes = [m.value for m in _SAM3_COMPRESSION_MODES]
    parser = argparse.ArgumentParser(
        description="Quantize SAM3 models for faster OpenVINO inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Compression modes:
  {', '.join(valid_modes)}
  all         Run all compression modes

Examples:
  python quantize_sam3_openvino.py --compression int8_sym --source-dir ./sam3-openvino/openvino-fp16
  python quantize_sam3_openvino.py --compression int4_sym --source-dir ./sam3-openvino/openvino-fp16
  python quantize_sam3_openvino.py --compression all --source-dir ./sam3-openvino/openvino-fp16 --validate
        """,
    )
    parser.add_argument(
        "--compression",
        type=str,
        required=True,
        choices=[*valid_modes, "all"],
        help="Weight compression mode to apply.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./sam3-openvino"),
        help="Base output directory. Default: ./sam3-openvino",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Directory with FP16 OpenVINO IR models.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate models after conversion.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="OpenVINO device for validation. Default: CPU",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print comparison table of all variants in output-dir.",
    )
    return parser


def main() -> None:
    """CLI entry point for SAM3 quantization."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

    if args.source_dir is None:
        parser.error("--source-dir is required. Point it to your FP16 OpenVINO IR directory.")

    # Resolve compression modes to run
    modes = _SAM3_COMPRESSION_MODES if args.compression == "all" else [CompressionMode(args.compression)]

    # Run each compression mode
    result_dirs: dict[str, Path] = {}
    for mode in modes:
        try:
            result_dirs[mode.value] = apply_weight_compression(args.source_dir, args.output_dir, mode)
        except Exception:  # noqa: PERF203
            logger.exception("Failed compression: %s", mode.value)

    # Validate
    if args.validate:
        for name, result_dir in result_dirs.items():
            logger.info("-" * 60)
            logger.info("Validating: %s", name)
            validate_openvino_models(result_dir, device=args.device)

    # Summary
    logger.info("=" * 60)
    logger.info("Quantization complete!")
    for name, result_dir in result_dirs.items():
        size = get_dir_size(result_dir)
        logger.info("  %s: %s (%.1f MB model files)", name, result_dir, size)

    # Print comparison table
    if args.compare or args.compression == "all":
        print_comparison_table(args.output_dir)


if __name__ == "__main__":
    main()
