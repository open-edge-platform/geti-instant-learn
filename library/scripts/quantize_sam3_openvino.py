# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Quantize SAM3 models to INT8/INT4 for faster inference with OpenVINO.

Supports two quantization paths:

**Path A — Pre-quantized ONNX from usls:**
  Download Q8, Q4F16, or BNB4 ONNX models from the usls project.
  These are kept as ONNX files (OpenVINO loads them directly).
  Converting quantized ONNX to IR inflates weights, so ONNX is preferred.

**Path B — NNCF weight compression:**
  Apply INT8 or INT4 weight compression to existing FP16 OpenVINO IR models
  using OpenVINO's NNCF framework. Produces proper OpenVINO IR with
  compressed weights. No calibration data needed.

Usage:
    # Download Q8 ONNX models (kept as ONNX — ~845 MB)
    python quantize_sam3_openvino.py --method q8 --output-dir ./sam3-openvino

    # Download Q4F16 ONNX models (~564 MB)
    python quantize_sam3_openvino.py --method q4f16 --output-dir ./sam3-openvino

    # Download BNB4 ONNX models (~688 MB)
    python quantize_sam3_openvino.py --method bnb4 --output-dir ./sam3-openvino

    # Apply NNCF INT8 weight compression to FP16 IR → proper IR output
    python quantize_sam3_openvino.py --method nncf-int8 --source-dir ./sam3-openvino/openvino-fp16

    # Apply NNCF INT4 weight compression to FP16 IR → proper IR output
    python quantize_sam3_openvino.py --method nncf-int4 --source-dir ./sam3-openvino/openvino-fp16

    # Download all usls variants at once
    python quantize_sam3_openvino.py --method all-usls --output-dir ./sam3-openvino

    # Run all methods (usls + NNCF) and compare sizes
    python quantize_sam3_openvino.py --method all --source-dir ./sam3-openvino/openvino-fp16

    # Validate quantized models
    python quantize_sam3_openvino.py --method q8 --validate
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import openvino as ov
import requests
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from rich.table import Table

logger = logging.getLogger(__name__)

# Pre-exported ONNX model URLs from usls project (v2 split - 3 models)
ONNX_BASE_URL = "https://github.com/jamjamjon/assets/releases/download/sam3"

# Canonical v2 model names
MODEL_NAMES = [
    "vision-encoder",
    "text-encoder",
    "geo-encoder-mask-decoder",
]

# ONNX file suffixes for each quantization type
ONNX_SUFFIXES: dict[str, str] = {
    "fp32": "",
    "fp16": "-fp16",
    "q8": "-q8",
    "q4f16": "-q4f16",
    "bnb4": "-bnb4",
}

# Methods that download from usls
USLS_METHODS = {"q8", "q4f16", "bnb4"}

# Methods that use NNCF
NNCF_METHODS = {"nncf-int8", "nncf-int4"}

# All individual methods
ALL_METHODS = USLS_METHODS | NNCF_METHODS

# Tokenizer files needed for inference
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]


def download_file(url: str, target_path: Path) -> None:
    """Download a file from a URL with progress bar.

    Args:
        url: URL to download from.
        target_path: Local path to save the file.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)

    disable_progress = not sys.stderr.isatty()
    progress = Progress(
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        " • ",
        DownloadColumn(),
        " • ",
        TransferSpeedColumn(),
        " • ",
        TimeRemainingColumn(),
        transient=True,
        disable=disable_progress,
    )

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        msg = f"Downloading {target_path.name} ({total_size / (1024 * 1024):.1f} MB)..."
        logger.info(msg)

        with progress:
            task_id = progress.add_task("download", total=total_size, filename=target_path.name)
            with target_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress.update(task_id, advance=len(chunk))

    msg = f"Downloaded: {target_path}"
    logger.info(msg)


def download_quantized_onnx(quant_type: str, output_dir: Path) -> Path:
    """Download pre-quantized SAM3 ONNX models from usls.

    Args:
        quant_type: Quantization type ("q8", "q4f16", or "bnb4").
        output_dir: Base output directory.

    Returns:
        Path to directory containing downloaded ONNX models.
    """
    suffix = ONNX_SUFFIXES[quant_type]
    onnx_dir = output_dir / f"onnx-v2-{quant_type}"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    for model_name in MODEL_NAMES:
        filename = f"{model_name}{suffix}.onnx"
        target_path = onnx_dir / filename
        if target_path.exists():
            msg = f"Already exists, skipping: {target_path}"
            logger.info(msg)
            continue
        url = f"{ONNX_BASE_URL}/{filename}"
        download_file(url, target_path)

    # Download tokenizer files
    for filename in TOKENIZER_FILES:
        target_path = onnx_dir / filename
        if target_path.exists():
            continue
        url = f"{ONNX_BASE_URL}/{filename}"
        try:
            download_file(url, target_path)
        except requests.HTTPError:
            msg = f"Could not download {filename} (optional)"
            logger.warning(msg)

    return onnx_dir


def organize_usls_quantized(quant_type: str, output_dir: Path) -> Path:
    """Download pre-quantized ONNX from usls and organize with canonical names.

    Quantized ONNX files are kept as ONNX rather than converted to OpenVINO IR,
    because ``ov.convert_model()`` decompresses quantized weights, inflating the
    model size (e.g., Q8 ONNX ~845 MB becomes ~3.1 GB as IR). OpenVINO can load
    ONNX files directly, so no conversion is needed.

    Files are copied with canonical names (e.g., ``vision-encoder.onnx``) so that
    ``SAM3OpenVINO`` can find them using the standard search pattern.

    Args:
        quant_type: Quantization type ("q8", "q4f16", or "bnb4").
        output_dir: Base output directory.

    Returns:
        Path to directory containing organized ONNX models.
    """
    logger.info("=" * 60)
    logger.info("Processing usls pre-quantized variant: %s", quant_type)
    logger.info("=" * 60)

    onnx_dir = download_quantized_onnx(quant_type, output_dir)

    # Output directory uses onnx- prefix to make it clear these are ONNX files
    out_dir = output_dir / f"onnx-{quant_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = ONNX_SUFFIXES[quant_type]
    for model_name in MODEL_NAMES:
        src = onnx_dir / f"{model_name}{suffix}.onnx"
        # Canonical name so SAM3OpenVINO finds it as {name}.onnx
        dst = out_dir / f"{model_name}.onnx"
        if dst.exists():
            size_mb = dst.stat().st_size / (1024 * 1024)
            logger.info("Already exists: %s (%.1f MB)", dst, size_mb)
            continue
        if not src.exists():
            logger.error("ONNX file not found: %s", src)
            continue
        shutil.copy2(src, dst)
        size_mb = dst.stat().st_size / (1024 * 1024)
        logger.info("Copied: %s → %s (%.1f MB)", src.name, dst, size_mb)

    # Copy tokenizer files
    for filename in TOKENIZER_FILES:
        src = onnx_dir / filename
        dst = out_dir / filename
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    return out_dir


def _compress_single_model(nncf: object, ov_model: ov.Model, mode: str) -> ov.Model:
    """Apply NNCF weight compression to a single OpenVINO model.

    For INT4 mode, attempts group_size=128 first. If that fails due to
    incompatible channel sizes, retries with group_size=-1 (per-channel).

    Args:
        nncf: The imported nncf module.
        ov_model: OpenVINO model to compress.
        mode: Compression mode ("int8" or "int4").

    Returns:
        Compressed OpenVINO model.
    """
    if mode == "int8":
        return nncf.compress_weights(
            ov_model,
            mode=nncf.CompressWeightsMode.INT8_SYM,
        )
    # INT4: try group_size=128 first, fall back to per-channel
    try:
        return nncf.compress_weights(
            ov_model,
            mode=nncf.CompressWeightsMode.INT4_SYM,
            ratio=0.8,
            group_size=128,
        )
    except nncf.errors.InvalidGroupSizeError:
        logger.warning("  group_size=128 failed, retrying with per-channel (group_size=-1)...")
        return nncf.compress_weights(
            ov_model,
            mode=nncf.CompressWeightsMode.INT4_SYM,
            ratio=0.8,
            group_size=-1,
        )


def apply_nncf_weight_compression(
    source_dir: Path,
    output_dir: Path,
    mode: str = "int8",
) -> Path:
    """Apply NNCF weight compression to OpenVINO IR models.

    Args:
        source_dir: Directory containing FP16 OpenVINO IR models.
        output_dir: Base output directory.
        mode: Compression mode ("int8" or "int4").

    Returns:
        Path to directory containing compressed OpenVINO IR models.

    Raises:
        ImportError: If NNCF is not installed.
        ValueError: If an unknown compression mode is specified.
    """
    try:
        import nncf  # noqa: PLC0415
    except ImportError:
        msg = "nncf is required for weight compression. Install it with: uv pip install nncf"
        raise ImportError(msg) from None

    if mode not in {"int8", "int4"}:
        msg = f"Unknown NNCF mode: {mode}"
        raise ValueError(msg)

    logger.info("=" * 60)
    logger.info("Applying NNCF %s weight compression", mode.upper())
    logger.info("Using nncf version: %s", nncf.__version__)
    logger.info("=" * 60)

    ir_dir = output_dir / f"openvino-nncf-{mode}"
    ir_dir.mkdir(parents=True, exist_ok=True)

    core = ov.Core()

    for model_name in MODEL_NAMES:
        xml_path = source_dir / f"{model_name}.xml"
        if not xml_path.exists():
            msg = f"Source model not found: {xml_path}"
            logger.warning(msg)
            continue

        msg = f"Compressing {model_name} with NNCF {mode.upper()}..."
        logger.info(msg)

        ov_model = core.read_model(xml_path)

        try:
            compressed_model = _compress_single_model(nncf, ov_model, mode)
        except Exception:
            logger.exception("Failed to compress %s with NNCF %s", model_name, mode)
            continue

        out_xml = ir_dir / f"{model_name}.xml"
        ov.save_model(compressed_model, out_xml)

        bin_path = ir_dir / f"{model_name}.bin"
        size_mb = bin_path.stat().st_size / (1024 * 1024)
        msg = f"Saved: {out_xml} ({size_mb:.1f} MB)"
        logger.info(msg)

    # Copy tokenizer files from source
    for filename in TOKENIZER_FILES:
        src = source_dir / filename
        dst = ir_dir / filename
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    return ir_dir


def validate_openvino_models(model_dir: Path, device: str = "CPU") -> bool:
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

    # Find model files (prefer .xml, fallback to .onnx)
    model_files = {}
    for model_name in MODEL_NAMES:
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

    # Validate decoder
    if "geo-encoder-mask-decoder" in model_files:
        try:
            model = core.compile_model(model_files["geo-encoder-mask-decoder"], device)
            result = model([
                rng.standard_normal((1, 256, 288, 288)).astype(np.float32),
                rng.standard_normal((1, 256, 144, 144)).astype(np.float32),
                rng.standard_normal((1, 256, 72, 72)).astype(np.float32),
                rng.standard_normal((1, 256, 72, 72)).astype(np.float32),
                rng.standard_normal((1, 32, 256)).astype(np.float32),
                np.ones((1, 32), dtype=bool),
                np.zeros((1, 1, 4), dtype=np.float32),
                np.full((1, 1), -10, dtype=np.int64),
            ])
            shapes = {
                name: result[name].shape for name in ["pred_masks", "pred_boxes", "pred_logits", "presence_logits"]
            }
            logger.info("  geo-encoder-mask-decoder: OK — %s", shapes)
        except Exception:
            logger.exception("  geo-encoder-mask-decoder: FAILED")
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
    table.add_column("Vision Enc.", justify="right")
    table.add_column("Text Enc.", justify="right")
    table.add_column("Decoder", justify="right")
    table.add_column("Total", justify="right", style="bold")
    table.add_column("Status", style="green")

    # Find all variant directories (openvino-* and onnx-*)
    variant_dirs = sorted([*output_dir.glob("openvino-*"), *output_dir.glob("onnx-*")])
    for variant_dir in variant_dirs:
        if not variant_dir.is_dir():
            continue
        variant_name = variant_dir.name
        for prefix in ("openvino-", "onnx-", "onnx-v2-"):
            variant_name = variant_name.replace(prefix, "", 1) if variant_name.startswith(prefix) else variant_name
        fmt = "IR" if variant_dir.name.startswith("openvino") else "ONNX"
        variant_label = f"{variant_name} ({fmt})"

        sizes = {}
        for model_name in MODEL_NAMES:
            bin_path = variant_dir / f"{model_name}.bin"
            if bin_path.exists():
                sizes[model_name] = bin_path.stat().st_size / (1024 * 1024)
            else:
                # Check for onnx fallback
                onnx_files = list(variant_dir.glob(f"{model_name}*.onnx"))
                if onnx_files:
                    sizes[model_name] = onnx_files[0].stat().st_size / (1024 * 1024)

        total = sum(sizes.values())
        has_all = len(sizes) == 3
        status = "OK" if has_all else f"Missing {3 - len(sizes)} model(s)"

        table.add_row(
            variant_label,
            f"{sizes.get('vision-encoder', 0):.1f} MB",
            f"{sizes.get('text-encoder', 0):.1f} MB",
            f"{sizes.get('geo-encoder-mask-decoder', 0):.1f} MB",
            f"{total:.1f} MB",
            status,
        )

    console.print(table)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the quantization CLI.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Quantize SAM3 models for faster OpenVINO inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
  q8          Download Q8 ONNX from usls, kept as ONNX (~845 MB)
  q4f16       Download Q4F16 ONNX from usls, kept as ONNX (~564 MB)
  bnb4        Download BNB4 ONNX from usls, kept as ONNX (~688 MB)
  nncf-int8   Apply NNCF INT8 weight compression to FP16 models (requires --source-dir)
  nncf-int4   Apply NNCF INT4 weight compression to FP16 models (requires --source-dir)
  all-usls    Download and convert all usls variants (q8 + q4f16 + bnb4)
  all         Run all methods (usls + NNCF, requires --source-dir for NNCF)

Examples:
  python quantize_sam3_openvino.py --method q8 --validate
  python quantize_sam3_openvino.py --method all-usls --output-dir ./sam3-openvino
  python quantize_sam3_openvino.py --method nncf-int8 --source-dir ./sam3-openvino/openvino-fp16
  python quantize_sam3_openvino.py --method all --source-dir ./sam3-openvino/openvino-fp16 --validate
        """,
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["q8", "q4f16", "bnb4", "nncf-int8", "nncf-int4", "all-usls", "all"],
        help="Quantization method to apply.",
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
        help="Directory with FP16 OpenVINO IR models (required for NNCF methods).",
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


def _run_method(method: str, output_dir: Path, source_dir: Path | None) -> Path:
    """Run a single quantization method.

    Args:
        method: Method name (e.g., "q8", "nncf-int8").
        output_dir: Base output directory.
        source_dir: Source directory for NNCF methods.

    Returns:
        Path to directory containing the quantized models.
    """
    if method in USLS_METHODS:
        return organize_usls_quantized(method, output_dir)
    # NNCF method
    nncf_mode = method.replace("nncf-", "")
    return apply_nncf_weight_compression(source_dir, output_dir, nncf_mode)


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

    method = args.method
    output_dir = args.output_dir
    source_dir = args.source_dir

    # Determine which methods to run
    methods_to_run = _resolve_methods(method)

    # Validate that source-dir is provided for NNCF
    nncf_needed = any(m in NNCF_METHODS for m in methods_to_run)
    if nncf_needed and source_dir is None:
        parser.error("--source-dir is required for NNCF methods. Point it to your FP16 OpenVINO IR directory.")

    # Run each method
    result_dirs: dict[str, Path] = {}
    for m in methods_to_run:
        try:
            result_dirs[m] = _run_method(m, output_dir, source_dir)
        except Exception:  # noqa: PERF203
            logger.exception("Failed method: %s", m)

    # Validate
    if args.validate:
        for m, result_dir in result_dirs.items():
            logger.info("-" * 60)
            logger.info("Validating: %s", m)
            validate_openvino_models(result_dir, device=args.device)

    # Summary
    logger.info("=" * 60)
    logger.info("Quantization complete!")
    for m, result_dir in result_dirs.items():
        size = get_dir_size(result_dir)
        logger.info("  %s: %s (%.1f MB model files)", m, result_dir, size)

    # Print comparison table
    if args.compare or method in {"all-usls", "all"}:
        print_comparison_table(output_dir)


def _resolve_methods(method: str) -> list[str]:
    """Resolve a method argument to a list of individual methods.

    Args:
        method: Method string from CLI (may be "all-usls", "all", or a single method).

    Returns:
        List of individual method names.
    """
    if method == "all-usls":
        return sorted(USLS_METHODS)
    if method == "all":
        return sorted(ALL_METHODS)
    return [method]


if __name__ == "__main__":
    main()
