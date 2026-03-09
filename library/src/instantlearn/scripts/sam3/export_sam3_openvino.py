# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export SAM3 PyTorch model to OpenVINO IR via ONNX.

Loads the official ``Sam3Model`` weights from HuggingFace (or a local
checkpoint), exports each sub-component to ONNX, then converts to OpenVINO IR.

Produces 5 OpenVINO IR models (4-model split + exemplar variant):

* ``vision-encoder``   — ViT + FPN backbone
* ``text-encoder``     — CLIP text encoder + projection
* ``geometry-encoder`` — Geometry encoder (classic, ``drop_spatial_bias=False``)
* ``geometry-encoder-exemplar`` — Geometry encoder (exemplar, ``drop_spatial_bias=True``)
* ``prompt-decoder``   — DETR encoder/decoder + box head + scoring + mask decoder

Usage:
    # Export from HuggingFace (default: facebook/sam3) to FP16 OpenVINO IR
    python export_sam3_openvino.py --output-dir ./sam3-openvino

    # Export with FP32 precision
    python export_sam3_openvino.py --output-dir ./sam3-openvino --precision fp32

    # Export from a local checkpoint
    python export_sam3_openvino.py --model-id /path/to/sam3.pt --output-dir ./sam3-openvino

    # Export and validate with dummy inference
    python export_sam3_openvino.py --output-dir ./sam3-openvino --validate

    # Only convert existing ONNX models to OpenVINO IR (skip PyTorch export)
    python export_sam3_openvino.py --onnx-dir ./sam3-onnx --output-dir ./sam3-openvino --skip-export
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def export_from_pytorch(
    model_id: str,
    output_dir: Path,
    *,
    resolution: int = 1008,
    opset_version: int = 17,
) -> dict[str, Path]:
    """Load Sam3Model and export all sub-components to ONNX.

    Args:
        model_id: HuggingFace model ID or local path to ``sam3.pt``.
        output_dir: Directory to write the ONNX files and tokenizer.
        resolution: Input image resolution.
        opset_version: ONNX opset version.

    Returns:
        Mapping from model name to ONNX file path.
    """
    import torch  # noqa: PLC0415
    from transformers import CLIPTokenizerFast  # noqa: PLC0415

    from instantlearn.models.sam3.export_openvino import export_sam3_to_onnx  # noqa: PLC0415
    from instantlearn.models.sam3.model import Sam3Model  # noqa: PLC0415

    logger.info("Loading Sam3Model from '%s'...", model_id)
    model = Sam3Model.from_pretrained(model_id, device="cpu", dtype=torch.float32)
    model.eval()
    logger.info("Model loaded successfully.")

    # Export ONNX
    onnx_dir = output_dir / "onnx"
    with torch.no_grad():
        exported = export_sam3_to_onnx(
            model,
            onnx_dir,
            resolution=resolution,
            opset_version=opset_version,
        )

    # Save tokenizer alongside the ONNX files
    logger.info("Saving tokenizer...")
    tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
    tokenizer.save_pretrained(str(onnx_dir))
    logger.info("  Tokenizer saved to %s", onnx_dir)

    return exported


def convert_to_openvino(
    onnx_dir: Path,
    output_dir: Path,
    *,
    precision: str = "fp16",
) -> dict[str, Path]:
    """Convert all SAM3 ONNX models to OpenVINO IR.

    Args:
        onnx_dir: Directory containing ONNX files from ``export_from_pytorch``.
        output_dir: Directory to write OpenVINO IR files.
        precision: Target precision (``"fp32"`` or ``"fp16"``).

    Returns:
        Mapping from model name to ``.xml`` path.
    """
    from instantlearn.models.sam3.export_openvino import convert_onnx_to_openvino  # noqa: PLC0415

    compress_to_fp16 = precision == "fp16"
    converted = convert_onnx_to_openvino(
        onnx_dir,
        output_dir,
        compress_to_fp16=compress_to_fp16,
    )

    # Copy tokenizer files to the IR output directory
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    for filename in tokenizer_files:
        src = onnx_dir / filename
        dst = output_dir / filename
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            logger.info("Copied tokenizer file: %s", filename)

    return converted


def validate_openvino_models(  # noqa: PLR0915
    model_dir: Path,
    device: str = "CPU",
    resolution: int = 1008,
) -> None:
    """Validate exported OpenVINO models with dummy inference.

    Args:
        model_dir: Directory containing OpenVINO IR files.
        device: OpenVINO device for validation.
        resolution: Model input resolution.
    """
    import openvino as ov  # noqa: PLC0415

    core = ov.Core()
    rng = np.random.default_rng(42)
    feat_size = resolution // 14  # 72 for 1008

    logger.info("Validating OpenVINO models in %s...", model_dir)

    # 1. Vision encoder
    vision_xml = model_dir / "vision-encoder.xml"
    if vision_xml.exists():
        vision_model = core.compile_model(vision_xml, device)
        dummy_img = rng.standard_normal((1, 3, resolution, resolution)).astype(np.float32)
        vision_result = vision_model([dummy_img])
        logger.info(
            "  Vision encoder: OK — %s",
            {k: vision_result[k].shape for k in ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"]},
        )
    else:
        logger.warning("  Vision encoder: MISSING (%s)", vision_xml)

    # 2. Text encoder
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
    else:
        logger.warning("  Text encoder: MISSING (%s)", text_xml)

    # 3. Geometry encoder (classic)
    geo_xml = model_dir / "geometry-encoder.xml"
    if geo_xml.exists():
        geo_model = core.compile_model(geo_xml, device)
        dummy_fpn = rng.standard_normal((1, 256, feat_size, feat_size)).astype(np.float32)
        dummy_pos = rng.standard_normal((1, 256, feat_size, feat_size)).astype(np.float32)
        dummy_boxes = rng.random((1, 1, 4)).astype(np.float32)
        dummy_box_labels = np.ones((1, 1), dtype=np.int64)
        dummy_points = np.zeros((1, 1, 2), dtype=np.float32)
        dummy_point_labels = np.full((1, 1), -10, dtype=np.int64)
        geo_result = geo_model([dummy_fpn, dummy_pos, dummy_boxes, dummy_box_labels, dummy_points, dummy_point_labels])
        logger.info(
            "  Geometry encoder (classic): OK — %s",
            {k: geo_result[k].shape for k in ["geometry_features", "geometry_mask"]},
        )
    else:
        logger.warning("  Geometry encoder (classic): MISSING (%s)", geo_xml)

    # 4. Geometry encoder (exemplar)
    geo_ex_xml = model_dir / "geometry-encoder-exemplar.xml"
    if geo_ex_xml.exists():
        geo_ex_model = core.compile_model(geo_ex_xml, device)
        dummy_boxes_ign = np.zeros((1, 1, 4), dtype=np.float32)
        dummy_box_labels_ign = np.full((1, 1), -10, dtype=np.int64)
        dummy_pts = rng.random((1, 1, 2)).astype(np.float32)
        dummy_pt_labels = np.ones((1, 1), dtype=np.int64)
        geo_ex_result = geo_ex_model([
            dummy_fpn,
            dummy_pos,
            dummy_boxes_ign,
            dummy_box_labels_ign,
            dummy_pts,
            dummy_pt_labels,
        ])
        logger.info(
            "  Geometry encoder (exemplar): OK — %s",
            {k: geo_ex_result[k].shape for k in ["geometry_features", "geometry_mask"]},
        )
    else:
        logger.warning("  Geometry encoder (exemplar): MISSING (%s)", geo_ex_xml)

    # 5. Prompt decoder
    dec_xml = model_dir / "prompt-decoder.xml"
    if dec_xml.exists():
        dec_model = core.compile_model(dec_xml, device)
        dummy_f0 = rng.standard_normal((1, 256, feat_size * 4, feat_size * 4)).astype(np.float32)
        dummy_f1 = rng.standard_normal((1, 256, feat_size * 2, feat_size * 2)).astype(np.float32)
        dummy_f2 = rng.standard_normal((1, 256, feat_size, feat_size)).astype(np.float32)
        dummy_p2 = rng.standard_normal((1, 256, feat_size, feat_size)).astype(np.float32)
        dummy_prompt = rng.standard_normal((1, 32, 256)).astype(np.float32)
        dummy_pmask = np.ones((1, 32), dtype=bool)
        dec_result = dec_model([dummy_f0, dummy_f1, dummy_f2, dummy_p2, dummy_prompt, dummy_pmask])
        logger.info(
            "  Prompt decoder: OK — %s",
            {k: dec_result[k].shape for k in ["pred_masks", "pred_boxes", "pred_logits", "presence_logits"]},
        )
    else:
        logger.warning("  Prompt decoder: MISSING (%s)", dec_xml)

    logger.info("Validation complete!")


def main() -> None:
    """CLI entry point for SAM3 PyTorch → ONNX → OpenVINO export."""
    parser = argparse.ArgumentParser(
        description="Export SAM3 PyTorch model to OpenVINO IR via ONNX (4-model split).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: PyTorch → ONNX → OpenVINO IR (FP16)
  python export_sam3_openvino.py --output-dir ./sam3-openvino

  # FP32 precision
  python export_sam3_openvino.py --output-dir ./sam3-openvino --precision fp32

  # From local checkpoint
  python export_sam3_openvino.py --model-id /path/to/sam3.pt --output-dir ./sam3-openvino

  # Only convert existing ONNX to OpenVINO IR
  python export_sam3_openvino.py --onnx-dir ./sam3-onnx --output-dir ./sam3-openvino --skip-export

  # Export and validate
  python export_sam3_openvino.py --output-dir ./sam3-openvino --validate
        """,
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/sam3",
        help="HuggingFace model ID or local path. Default: facebook/sam3",
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
        "--onnx-dir",
        type=Path,
        default=None,
        help="Path to existing ONNX directory (skip PyTorch export).",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip PyTorch → ONNX export and only convert existing ONNX files.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1008,
        help="Input image resolution. Default: 1008",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version. Default: 17",
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # Step 1: PyTorch → ONNX export
    if args.skip_export:
        if args.onnx_dir is None:
            parser.error("--onnx-dir is required when using --skip-export")
        onnx_dir = args.onnx_dir
        logger.info("Skipping PyTorch export; using ONNX models from %s", onnx_dir)
    else:
        exported = export_from_pytorch(
            model_id=args.model_id,
            output_dir=args.output_dir,
            resolution=args.resolution,
            opset_version=args.opset_version,
        )
        onnx_dir = args.output_dir / "onnx"
        logger.info("ONNX export complete: %s", list(exported.keys()))

    # Step 2: ONNX → OpenVINO IR
    logger.info("Converting ONNX → OpenVINO IR (precision=%s)...", args.precision)
    converted = convert_to_openvino(
        onnx_dir=onnx_dir,
        output_dir=args.output_dir,
        precision=args.precision,
    )
    logger.info("OpenVINO conversion complete!")
    for name, path in converted.items():
        logger.info("  %s: %s", name, path)

    # Step 3: Validate
    if args.validate:
        validate_openvino_models(
            args.output_dir,
            device=args.device,
            resolution=args.resolution,
        )

    logger.info("Done! Models saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
