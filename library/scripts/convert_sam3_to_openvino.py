# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Convert SAM3 ONNX models to OpenVINO IR format.

Expects a 5-model split (exported via ``export_sam3_openvino.py``)::

    vision-encoder.onnx
    text-encoder.onnx
    geometry-encoder.onnx
    geometry-encoder-exemplar.onnx
    prompt-decoder.onnx

Usage:
    # Convert local ONNX models to OpenVINO IR
    python convert_sam3_to_openvino.py --onnx-dir ./onnx-models --output-dir ./openvino-models

    # Convert with FP16 compression
    python convert_sam3_to_openvino.py --onnx-dir ./onnx-models --output-dir ./openvino-models --precision fp16

Note:
    To export from PyTorch, use ``export_sam3_openvino.py`` first.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import openvino as ov

logger = logging.getLogger(__name__)

# Canonical model names (5-model split from export_sam3_openvino.py)
OV_MODEL_NAMES = [
    "vision-encoder",
    "text-encoder",
    "geometry-encoder",
    "geometry-encoder-exemplar",
    "prompt-decoder",
]


def _find_onnx_file(onnx_dir: Path, name: str) -> Path | None:
    """Find an ONNX file by base name, trying fp32 then fp16 variants.

    Args:
        onnx_dir: Directory to search.
        name: Base model name (without extension).

    Returns:
        Path to found file, or ``None``.
    """
    for suffix in (f"{name}.onnx", f"{name}-fp16.onnx"):
        candidate = onnx_dir / suffix
        if candidate.exists():
            return candidate
    return None


def find_onnx_models(onnx_dir: Path) -> dict[str, Path]:
    """Find ONNX model files in a directory.

    Expects 5 models: vision-encoder, text-encoder, geometry-encoder,
    geometry-encoder-exemplar, prompt-decoder.

    Args:
        onnx_dir: Directory containing ONNX models.

    Returns:
        Dictionary mapping canonical name to ONNX file path.

    Raises:
        FileNotFoundError: If required model files are missing.
    """
    models = {}
    for name in OV_MODEL_NAMES:
        path = _find_onnx_file(onnx_dir, name)
        if path is not None:
            models[name] = path

    if len(models) == len(OV_MODEL_NAMES):
        logger.info("Found all %d ONNX models.", len(models))
        return models

    missing = [n for n in OV_MODEL_NAMES if n not in models]
    msg = f"Incomplete model set in {onnx_dir}. Missing: {missing}. Expected all of {OV_MODEL_NAMES}."
    raise FileNotFoundError(msg)


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
    """Convert all SAM3 ONNX models to OpenVINO IR format.

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

    logger.info("Validating OpenVINO models in %s ...", model_dir)

    # Vision encoder
    vision_xml = model_dir / "vision-encoder.xml"
    if vision_xml.exists():
        vision_model = core.compile_model(vision_xml, device)
        dummy_image = rng.standard_normal((1, 3, 1008, 1008)).astype(np.float32)
        vision_result = vision_model([dummy_image])
        logger.info(
            "  Vision encoder: OK — %s",
            {k: vision_result[k].shape for k in ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"]},
        )

    # Text encoder
    text_xml = model_dir / "text-encoder.xml"
    if text_xml.exists():
        text_model = core.compile_model(text_xml, device)
        dummy_ids = np.ones((1, 32), dtype=np.int64)
        dummy_mask = np.ones((1, 32), dtype=np.int64)
        text_result = text_model([dummy_ids, dummy_mask])
        logger.info(
            "  Text encoder: OK — %s",
            {k: text_result[k].shape for k in ["text_features", "text_mask"]},
        )

    # Shared FPN dummies for decoder / geometry encoder validation
    dummy_fpn0 = rng.standard_normal((1, 256, 288, 288)).astype(np.float32)
    dummy_fpn1 = rng.standard_normal((1, 256, 144, 144)).astype(np.float32)
    dummy_fpn2 = rng.standard_normal((1, 256, 72, 72)).astype(np.float32)
    dummy_pos2 = rng.standard_normal((1, 256, 72, 72)).astype(np.float32)

    # Geometry encoder (classic)
    geo_xml = model_dir / "geometry-encoder.xml"
    if geo_xml.exists():
        geo_model = core.compile_model(geo_xml, device)
        dummy_boxes = rng.random((1, 1, 4)).astype(np.float32)
        dummy_box_labels = np.ones((1, 1), dtype=np.int64)
        dummy_pts = np.zeros((1, 1, 2), dtype=np.float32)
        dummy_pt_labels = np.full((1, 1), -10, dtype=np.int64)
        geo_result = geo_model([dummy_fpn2, dummy_pos2, dummy_boxes, dummy_box_labels, dummy_pts, dummy_pt_labels])
        logger.info(
            "  Geometry encoder (classic): OK — %s",
            {k: geo_result[k].shape for k in ["geometry_features", "geometry_mask"]},
        )

    # Geometry encoder (exemplar)
    geo_ex_xml = model_dir / "geometry-encoder-exemplar.xml"
    if geo_ex_xml.exists():
        geo_ex_model = core.compile_model(geo_ex_xml, device)
        dummy_boxes = np.zeros((1, 1, 4), dtype=np.float32)
        dummy_box_labels = np.full((1, 1), -10, dtype=np.int64)
        dummy_pts = rng.random((1, 1, 2)).astype(np.float32)
        dummy_pt_labels = np.ones((1, 1), dtype=np.int64)
        geo_ex_result = geo_ex_model([
            dummy_fpn2,
            dummy_pos2,
            dummy_boxes,
            dummy_box_labels,
            dummy_pts,
            dummy_pt_labels,
        ])
        logger.info(
            "  Geometry encoder (exemplar): OK — %s",
            {k: geo_ex_result[k].shape for k in ["geometry_features", "geometry_mask"]},
        )

    # Prompt decoder
    dec_xml = model_dir / "prompt-decoder.xml"
    if dec_xml.exists():
        dummy_prompt = rng.standard_normal((1, 32, 256)).astype(np.float32)
        dummy_pmask = np.ones((1, 32), dtype=bool)
        dec_model = core.compile_model(dec_xml, device)
        dec_result = dec_model([dummy_fpn0, dummy_fpn1, dummy_fpn2, dummy_pos2, dummy_prompt, dummy_pmask])
        logger.info(
            "  Prompt decoder: OK — %s",
            {k: dec_result[k].shape for k in ["pred_masks", "pred_boxes", "pred_logits", "presence_logits"]},
        )

    logger.info("Validation complete!")


def main() -> None:
    """CLI entry point for SAM3 ONNX → OpenVINO conversion."""
    parser = argparse.ArgumentParser(
        description="Convert SAM3 ONNX models to OpenVINO IR format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert local ONNX models
  python convert_sam3_to_openvino.py --onnx-dir ./onnx-models --output-dir ./openvino-models

  # Convert with FP16 compression
  python convert_sam3_to_openvino.py --onnx-dir ./onnx-models --output-dir ./openvino-models --precision fp16
        """,
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        required=True,
        help="Directory containing ONNX models (5-model split from export_sam3_openvino.py).",
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

    onnx_dir = args.onnx_dir

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
