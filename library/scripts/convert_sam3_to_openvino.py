# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Convert SAM3 ONNX models (v2 split) to OpenVINO IR format.

This script downloads pre-exported SAM3 ONNX models from the usls project
(https://github.com/jamjamjon/assets/releases/tag/sam3) and converts them
to OpenVINO IR (.xml/.bin) format for use with SAM3OpenVINO.

The v2 model split consists of 3 ONNX models:
    - vision-encoder.onnx: ViT + FPN backbone
    - text-encoder.onnx: CLIP text encoder + projection
    - geo-encoder-mask-decoder.onnx: Geometry encoder + DETR + mask decoder

Usage:
    # Convert local ONNX models to OpenVINO IR (FP32)
    python convert_sam3_to_openvino.py --onnx-dir ./onnx-models-v2 --output-dir ./openvino-models

    # Convert with FP16 compression
    python convert_sam3_to_openvino.py --onnx-dir ./onnx-models-v2 --output-dir ./openvino-models --precision fp16

    # Download ONNX models first, then convert
    python convert_sam3_to_openvino.py --download --output-dir ./openvino-models

    # Download FP16 ONNX models, then convert
    python convert_sam3_to_openvino.py --download --onnx-precision fp16 --output-dir ./openvino-models
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import openvino as ov
import requests
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn

logger = logging.getLogger(__name__)

# Pre-exported ONNX model URLs from usls project (v2 split - 3 models)
ONNX_BASE_URL = "https://github.com/jamjamjon/assets/releases/download/sam3"

# v2 model filenames by precision
ONNX_MODEL_FILES = {
    "fp32": [
        "vision-encoder.onnx",
        "text-encoder.onnx",
        "geo-encoder-mask-decoder.onnx",
    ],
    "fp16": [
        "vision-encoder-fp16.onnx",
        "text-encoder-fp16.onnx",
        "geo-encoder-mask-decoder-fp16.onnx",
    ],
}

# Canonical names for OpenVINO IR output (always use these regardless of ONNX precision)
OV_MODEL_NAMES = [
    "vision-encoder",
    "text-encoder",
    "geo-encoder-mask-decoder",
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


def download_onnx_models(output_dir: Path, onnx_precision: str = "fp32") -> Path:
    """Download pre-exported SAM3 ONNX models (v2 split).

    Args:
        output_dir: Directory to save ONNX models.
        onnx_precision: ONNX model precision to download ("fp32" or "fp16").

    Returns:
        Path to directory containing downloaded ONNX models.
    """
    onnx_dir = output_dir / f"onnx-v2-{onnx_precision}"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    model_files = ONNX_MODEL_FILES[onnx_precision]

    for filename in model_files:
        target_path = onnx_dir / filename
        if target_path.exists():
            msg = f"Already exists, skipping: {target_path}"
            logger.info(msg)
            continue
        url = f"{ONNX_BASE_URL}/{filename}"
        download_file(url, target_path)

    # Also download tokenizer files needed for inference
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]
    for filename in tokenizer_files:
        target_path = onnx_dir / filename
        if target_path.exists():
            continue
        url = f"{ONNX_BASE_URL}/{filename}"
        try:
            download_file(url, target_path)
        except requests.HTTPError:
            msg = f"Could not download {filename} (optional tokenizer file)"
            logger.warning(msg)

    return onnx_dir


def find_onnx_models(onnx_dir: Path) -> dict[str, Path]:
    """Find the 3 v2 ONNX model files in a directory.

    Supports both FP32 and FP16 naming conventions.

    Args:
        onnx_dir: Directory containing ONNX models.

    Returns:
        Dictionary mapping canonical name to ONNX file path.

    Raises:
        FileNotFoundError: If any required model is not found.
    """
    models = {}
    for canonical_name in OV_MODEL_NAMES:
        # Try FP32 name first, then FP16
        candidates = [
            onnx_dir / f"{canonical_name}.onnx",
            onnx_dir / f"{canonical_name}-fp16.onnx",
        ]
        found = None
        for candidate in candidates:
            if candidate.exists():
                found = candidate
                break
        if found is None:
            msg = (
                f"Could not find ONNX model '{canonical_name}' in {onnx_dir}. "
                f"Expected one of: {[c.name for c in candidates]}"
            )
            raise FileNotFoundError(msg)
        models[canonical_name] = found

    return models


def convert_onnx_to_openvino(
    onnx_path: Path,
    output_dir: Path,
    model_name: str,
    precision: str = "fp32",
) -> Path:
    """Convert a single ONNX model to OpenVINO IR format.

    Args:
        onnx_path: Path to input ONNX model.
        output_dir: Directory to save OpenVINO IR files.
        model_name: Name for the output model (without extension).
        precision: Target precision ("fp32" or "fp16").

    Returns:
        Path to the saved OpenVINO XML file.
    """
    msg = f"Converting {onnx_path.name} → {model_name}.xml (precision={precision})..."
    logger.info(msg)

    ov_model = ov.convert_model(onnx_path)

    if precision == "fp16":
        ov.save_model(ov_model, output_dir / f"{model_name}.xml", compress_to_fp16=True)
    else:
        ov.save_model(ov_model, output_dir / f"{model_name}.xml", compress_to_fp16=False)

    xml_path = output_dir / f"{model_name}.xml"
    bin_path = output_dir / f"{model_name}.bin"
    msg = f"Saved: {xml_path} ({bin_path.stat().st_size / (1024 * 1024):.1f} MB)"
    logger.info(msg)

    return xml_path


def convert_all(
    onnx_dir: Path,
    output_dir: Path,
    precision: str = "fp32",
) -> dict[str, Path]:
    """Convert all 3 SAM3 v2 ONNX models to OpenVINO IR format.

    Args:
        onnx_dir: Directory containing ONNX models.
        output_dir: Directory to save OpenVINO IR files.
        precision: Target precision ("fp32" or "fp16").

    Returns:
        Dictionary mapping model name to OpenVINO XML path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_models = find_onnx_models(onnx_dir)
    ov_models = {}

    for canonical_name, onnx_path in onnx_models.items():
        xml_path = convert_onnx_to_openvino(
            onnx_path=onnx_path,
            output_dir=output_dir,
            model_name=canonical_name,
            precision=precision,
        )
        ov_models[canonical_name] = xml_path

    # Copy tokenizer files to output directory if they exist in onnx_dir
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]
    for filename in tokenizer_files:
        src = onnx_dir / filename
        dst = output_dir / filename
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            msg = f"Copied tokenizer file: {filename}"
            logger.info(msg)

    return ov_models


def validate_openvino_models(model_dir: Path, device: str = "CPU") -> None:
    """Validate that OpenVINO models can be loaded and run with dummy inputs.

    Args:
        model_dir: Directory containing OpenVINO IR files.
        device: OpenVINO device to validate on.
    """
    core = ov.Core()

    rng = np.random.default_rng(42)

    logger.info("Validating OpenVINO models...")

    # Test vision encoder
    vision_model = core.compile_model(model_dir / "vision-encoder.xml", device)
    dummy_image = rng.standard_normal((1, 3, 1008, 1008)).astype(np.float32)
    vision_result = vision_model([dummy_image])
    logger.info(
        "  Vision encoder: OK — outputs: %s",
        {name: vision_result[name].shape for name in ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"]},
    )

    # Test text encoder
    text_model = core.compile_model(model_dir / "text-encoder.xml", device)
    dummy_ids = np.ones((1, 32), dtype=np.int64)
    dummy_mask = np.ones((1, 32), dtype=np.int64)
    text_result = text_model([dummy_ids, dummy_mask])
    logger.info(
        "  Text encoder: OK — outputs: %s",
        {name: text_result[name].shape for name in ["text_features", "text_mask"]},
    )

    # Test decoder (geo + mask decoder)
    decoder_model = core.compile_model(model_dir / "geo-encoder-mask-decoder.xml", device)
    dummy_fpn0 = rng.standard_normal((1, 256, 288, 288)).astype(np.float32)
    dummy_fpn1 = rng.standard_normal((1, 256, 144, 144)).astype(np.float32)
    dummy_fpn2 = rng.standard_normal((1, 256, 72, 72)).astype(np.float32)
    dummy_pos2 = rng.standard_normal((1, 256, 72, 72)).astype(np.float32)
    dummy_text_feats = rng.standard_normal((1, 32, 256)).astype(np.float32)
    dummy_text_mask = np.ones((1, 32), dtype=bool)
    dummy_boxes = np.zeros((1, 1, 4), dtype=np.float32)
    dummy_box_labels = np.full((1, 1), -10, dtype=np.int64)

    decoder_result = decoder_model([
        dummy_fpn0,
        dummy_fpn1,
        dummy_fpn2,
        dummy_pos2,
        dummy_text_feats,
        dummy_text_mask,
        dummy_boxes,
        dummy_box_labels,
    ])
    logger.info(
        "  Decoder: OK — outputs: %s",
        {name: decoder_result[name].shape for name in ["pred_masks", "pred_boxes", "pred_logits", "presence_logits"]},
    )

    logger.info("All models validated successfully!")


def main() -> None:
    """CLI entry point for SAM3 ONNX → OpenVINO conversion."""
    parser = argparse.ArgumentParser(
        description="Convert SAM3 ONNX models (v2 split) to OpenVINO IR format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert local ONNX models
  python convert_sam3_to_openvino.py --onnx-dir ./onnx-models-v2 --output-dir ./openvino-models

  # Download and convert in one step
  python convert_sam3_to_openvino.py --download --output-dir ./openvino-models

  # Download FP16 ONNX and convert to FP16 IR
  python convert_sam3_to_openvino.py --download --onnx-precision fp16 --precision fp16 --output-dir ./openvino-models
        """,
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        help="Directory containing ONNX models (v2 split). Required if --download is not set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./sam3-openvino"),
        help="Output directory for OpenVINO IR models. Default: ./sam3-openvino",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16"],
        default="fp16",
        help="Target OpenVINO IR precision. Default: fp16",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download pre-exported ONNX models before converting.",
    )
    parser.add_argument(
        "--onnx-precision",
        type=str,
        choices=["fp32", "fp16"],
        default="fp16",
        help="ONNX model precision to download (only with --download). Default: fp16",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate converted models with dummy inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="OpenVINO device for validation. Default: CPU",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if args.download:
        onnx_dir = download_onnx_models(args.output_dir, args.onnx_precision)
    elif args.onnx_dir:
        onnx_dir = args.onnx_dir
    else:
        parser.error("Either --onnx-dir or --download is required.")

    ov_models = convert_all(
        onnx_dir=onnx_dir,
        output_dir=args.output_dir,
        precision=args.precision,
    )

    logger.info("Conversion complete! OpenVINO models saved to: %s", args.output_dir)
    for name, path in ov_models.items():
        logger.info("  %s: %s", name, path)

    if args.validate:
        validate_openvino_models(args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
