# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ONNX export wrappers for SAM3 sub-components.

Each wrapper class isolates a portion of ``Sam3Model`` into a standalone
``nn.Module`` with a clean ``forward()`` signature that can be exported
with ``torch.onnx.export()``.

The 5-model split is designed for maximum flexibility:

1. **Vision encoder** — ViT backbone + FPN neck.
2. **Text encoder** — CLIP text encoder + linear projection.
3. **Geometry encoder** — Encodes box/point prompts with vision-conditioned
   transformer layers.  Two export variants are produced (``drop_spatial_bias``
   traced as ``True`` or ``False``).
4. **Prompt decoder** — DETR encoder + decoder + box refinement + dot-product
   scoring + mask decoder.  Accepts *pre-concatenated* prompt features (text
   ± geometry) so that the ONNX graph contains no conditional logic.

Text and geometry feature concatenation is performed in Python at inference time
(inside ``SAM3OpenVINO``), enabling both *classic* and *visual-exemplar* modes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from .model import Sam3Model

logger = logging.getLogger(__name__)


class OnnxVisionEncoder(nn.Module):
    """ONNX-exportable wrapper around the SAM3 vision encoder.

    Outputs only the first 3 FPN feature maps (scales 4x, 2x, 1x) and the
    position encoding at the 1x level.  The 0.5x level is dropped -- it is only
    used by the DETR encoder which receives ``fpn_feat_2`` directly.

    Inputs:
        pixel_values: ``[B, 3, 1008, 1008]``

    Outputs (named):
        fpn_feat_0: ``[B, 256, 288, 288]``  (4x scale)
        fpn_feat_1: ``[B, 256, 144, 144]``  (2x scale)
        fpn_feat_2: ``[B, 256,  72,  72]``  (1x scale)
        fpn_pos_2:  ``[B, 256,  72,  72]``  (position encoding at 1x scale)
    """

    def __init__(self, model: Sam3Model) -> None:
        """Store reference to the SAM3 vision encoder."""
        super().__init__()
        self.vision_encoder = model.vision_encoder

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run vision encoder and return FPN outputs.

        Args:
            pixel_values: Input image ``[B, 3, H, W]``.

        Returns:
            Tuple of ``(fpn_feat_0, fpn_feat_1, fpn_feat_2, fpn_pos_2)``.
        """
        vision_outputs = self.vision_encoder(pixel_values)
        fpn_hidden = vision_outputs["fpn_hidden_states"]
        fpn_pos = vision_outputs["fpn_position_encoding"]
        # Drop last FPN level (0.5x scale) -- used only inside model.forward()
        # via fpn_hidden_states[:-1], fpn_position_encoding[:-1]
        return fpn_hidden[0], fpn_hidden[1], fpn_hidden[2], fpn_pos[2]


class OnnxTextEncoder(nn.Module):
    """ONNX-exportable wrapper around the CLIP text encoder + projection.

    Inputs:
        input_ids:      ``[B, 32]`` int64
        attention_mask: ``[B, 32]`` int64

    Outputs (named):
        text_features: ``[B, 32, 256]`` float32
        text_mask:     ``[B, 32]`` bool
    """

    def __init__(self, model: Sam3Model) -> None:
        """Store references to the CLIP text encoder and projection."""
        super().__init__()
        self.text_encoder = model.text_encoder
        self.text_projection = model.text_projection

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text tokens into projected features.

        Args:
            input_ids: Token IDs ``[B, seq_len]``.
            attention_mask: Attention mask ``[B, seq_len]``.

        Returns:
            Tuple of ``(text_features, text_mask)``.
        """
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_features = self.text_projection(text_outputs.last_hidden_state)
        text_mask = attention_mask.bool()
        return text_features, text_mask


class OnnxGeometryEncoder(nn.Module):
    """ONNX-exportable wrapper around the SAM3 geometry encoder.

    Accepts box and/or point prompts together with the 1x FPN features, runs
    the geometry encoder transformer, and returns encoded prompt features.

    Because ONNX doesn't support Python-level if/else branching across
    ``drop_spatial_bias``, two separate ONNX files are exported:

    * ``geometry-encoder.onnx``  (``drop_spatial_bias=False``) — classic mode
    * ``geometry-encoder-exemplar.onnx``  (``drop_spatial_bias=True``) — exemplar fit

    Inputs:
        fpn_feat_2:    ``[B, 256, 72, 72]``
        fpn_pos_2:     ``[B, 256, 72, 72]``
        input_boxes:       ``[B, N, 4]``  cxcywh normalised (or all-zero)
        input_boxes_labels: ``[B, N]``    int64  (1=pos, 0=neg, -10=ignore)
        input_points:       ``[B, M, 2]``  xy normalised (or all-zero)
        input_points_labels: ``[B, M]``   int64  (1=pos, 0=neg, -10=ignore)

    Outputs (named):
        geometry_features: ``[B, K, 256]``
        geometry_mask:     ``[B, K]`` bool
    """

    def __init__(self, model: Sam3Model, *, drop_spatial_bias: bool = False) -> None:
        """Store geometry encoder and spatial bias flag."""
        super().__init__()
        self.geometry_encoder = model.geometry_encoder
        self.drop_spatial_bias = drop_spatial_bias

    def forward(
        self,
        fpn_feat_2: torch.Tensor,
        fpn_pos_2: torch.Tensor,
        input_boxes: torch.Tensor,
        input_boxes_labels: torch.Tensor,
        input_points: torch.Tensor,
        input_points_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode geometric prompts.

        Args:
            fpn_feat_2: Vision features at 1x scale ``[B, 256, 72, 72]``.
            fpn_pos_2: Position encoding at 1x scale ``[B, 256, 72, 72]``.
            input_boxes: Box coordinates in cxcywh normalised ``[B, N, 4]``.
            input_boxes_labels: Box labels ``[B, N]`` (1=pos, 0=neg, -10=ignore).
            input_points: Point coordinates in normalised xy ``[B, M, 2]``.
            input_points_labels: Point labels ``[B, M]`` (1=pos, 0=neg, -10=ignore).

        Returns:
            Tuple of ``(geometry_features, geometry_mask)``.
        """
        img_feats = (fpn_feat_2,)
        img_pos_embeds = (fpn_pos_2,)

        # Always compute masks and labels from the label tensors.
        # Invalid entries (label == -10) get mask=False, which the geometry
        # encoder's attention mechanism uses to ignore them.
        # This avoids data-dependent branching that torch.export cannot handle.
        box_mask = input_boxes_labels != -10
        box_labels = torch.where(input_boxes_labels == -10, 0, input_boxes_labels)
        box_embeddings = input_boxes.to(dtype=fpn_feat_2.dtype)

        point_mask = input_points_labels != -10
        point_labels = torch.where(input_points_labels == -10, 0, input_points_labels)
        point_embeddings = input_points.to(dtype=fpn_feat_2.dtype)

        geometry_outputs = self.geometry_encoder(
            box_embeddings=box_embeddings,
            box_mask=box_mask,
            box_labels=box_labels,
            point_embeddings=point_embeddings,
            point_mask=point_mask,
            point_labels=point_labels,
            img_feats=img_feats,
            img_pos_embeds=img_pos_embeds,
            drop_spatial_bias=self.drop_spatial_bias,
        )

        return (
            geometry_outputs["last_hidden_state"],
            geometry_outputs["attention_mask"],
        )


class OnnxPromptDecoder(nn.Module):
    """ONNX-exportable wrapper for the DETR pipeline + mask decoder.

    Accepts pre-concatenated prompt features (text alone, or text + geometry)
    and FPN features, runs the DETR encoder/decoder, box refinement, scoring,
    and mask decoder.

    Inputs:
        fpn_feat_0: ``[B, 256, 288, 288]``
        fpn_feat_1: ``[B, 256, 144, 144]``
        fpn_feat_2: ``[B, 256,  72,  72]``
        fpn_pos_2:  ``[B, 256,  72,  72]``
        prompt_features: ``[B, T, 256]``  (T = text tokens ± geometry tokens)
        prompt_mask:     ``[B, T]``       bool

    Outputs (named):
        pred_masks:       ``[B, 200, H_mask, W_mask]``
        pred_boxes:       ``[B, 200, 4]`` xyxy normalised
        pred_logits:      ``[B, 200]``
        presence_logits:  ``[B, 1]``
    """

    def __init__(self, model: Sam3Model) -> None:
        """Store references to DETR encoder, decoder, mask decoder, and scoring."""
        super().__init__()
        self.detr_encoder = model.detr_encoder
        self.detr_decoder = model.detr_decoder
        self.mask_decoder = model.mask_decoder
        self.dot_product_scoring = model.dot_product_scoring

    def forward(
        self,
        fpn_feat_0: torch.Tensor,
        fpn_feat_1: torch.Tensor,
        fpn_feat_2: torch.Tensor,
        fpn_pos_2: torch.Tensor,
        prompt_features: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run DETR pipeline and mask decoder.

        Args:
            fpn_feat_0: FPN features at 4x scale.
            fpn_feat_1: FPN features at 2x scale.
            fpn_feat_2: FPN features at 1x scale.
            fpn_pos_2: Position encoding at 1x scale.
            prompt_features: Pre-concatenated prompt features ``[B, T, 256]``.
            prompt_mask: Prompt attention mask ``[B, T]``.

        Returns:
            Tuple of ``(pred_masks, pred_boxes, pred_logits, presence_logits)``.
        """
        from .common import box_cxcywh_to_xyxy, inverse_sigmoid  # noqa: PLC0415

        # DETR encoder -- uses only the 1x scale as the vision "level"
        encoder_outputs = self.detr_encoder(
            vision_features=[fpn_feat_2],
            text_features=prompt_features,
            vision_pos_embeds=[fpn_pos_2],
            text_mask=prompt_mask,
        )

        # DETR decoder
        decoder_outputs = self.detr_decoder(
            vision_features=encoder_outputs["last_hidden_state"],
            text_features=encoder_outputs["text_features"],
            vision_pos_encoding=encoder_outputs["pos_embeds_flattened"],
            text_mask=prompt_mask,
            spatial_shapes=encoder_outputs["spatial_shapes"],
        )

        # Box refinement (last layer)
        all_box_offsets = self.detr_decoder.box_head(
            decoder_outputs["intermediate_hidden_states"],
        )
        reference_boxes_inv_sig = inverse_sigmoid(decoder_outputs["reference_boxes"])
        all_pred_boxes_cxcywh = (reference_boxes_inv_sig + all_box_offsets).sigmoid()
        all_pred_boxes = box_cxcywh_to_xyxy(all_pred_boxes_cxcywh)

        # Dot-product scoring
        all_pred_logits = self.dot_product_scoring(
            decoder_hidden_states=decoder_outputs["intermediate_hidden_states"],
            text_features=encoder_outputs["text_features"],
            text_mask=prompt_mask,
        ).squeeze(-1)

        pred_logits = all_pred_logits[-1]
        pred_boxes = all_pred_boxes[-1]
        decoder_hidden_states = decoder_outputs["intermediate_hidden_states"][-1]
        presence_logits = decoder_outputs["presence_logits"][-1]

        # Mask decoder
        fpn_hidden_states = [fpn_feat_0, fpn_feat_1, fpn_feat_2]
        mask_outputs = self.mask_decoder(
            decoder_queries=decoder_hidden_states,
            backbone_features=fpn_hidden_states,
            encoder_hidden_states=encoder_outputs["last_hidden_state"],
            prompt_features=prompt_features,
            prompt_mask=prompt_mask,
        )

        return (
            mask_outputs["pred_masks"],
            pred_boxes,
            pred_logits,
            presence_logits,
        )


_VISION_ENCODER_NAME = "vision-encoder"
_TEXT_ENCODER_NAME = "text-encoder"
_GEOMETRY_ENCODER_NAME = "geometry-encoder"
_GEOMETRY_ENCODER_EXEMPLAR_NAME = "geometry-encoder-exemplar"
_PROMPT_DECODER_NAME = "prompt-decoder"


def export_sam3_to_onnx(  # noqa: PLR0915
    model: Sam3Model,
    output_dir: str | Path,
    *,
    resolution: int = 1008,
    opset_version: int = 17,
) -> dict[str, Path]:
    """Export all SAM3 sub-components to ONNX.

    Produces 5 ONNX files in *output_dir*:

    * ``vision-encoder.onnx``
    * ``text-encoder.onnx``
    * ``geometry-encoder.onnx``  (classic mode, ``drop_spatial_bias=False``)
    * ``geometry-encoder-exemplar.onnx``  (exemplar fit, ``drop_spatial_bias=True``)
    * ``prompt-decoder.onnx``

    Args:
        model: A loaded ``Sam3Model`` instance (on CPU, eval mode).
        output_dir: Directory to write the ONNX files into.
        resolution: Input image resolution (default ``1008``).
        opset_version: ONNX opset version (default ``17``).

    Returns:
        Mapping from model name to the written ONNX path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device

    exported: dict[str, Path] = {}

    # --- 1. Vision encoder ---
    logger.info("Exporting vision encoder...")
    vision_wrapper = OnnxVisionEncoder(model)
    vision_wrapper.eval()
    dummy_pixel = torch.randn(1, 3, resolution, resolution, device=device)
    vision_path = output_dir / f"{_VISION_ENCODER_NAME}.onnx"
    torch.onnx.export(
        vision_wrapper,
        (dummy_pixel,),
        str(vision_path),
        opset_version=opset_version,
        dynamo=False,
        input_names=["pixel_values"],
        output_names=["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"],
        dynamic_axes={"pixel_values": {0: "batch"}},
    )
    exported[_VISION_ENCODER_NAME] = vision_path
    logger.info("  -> %s", vision_path)

    # --- 2. Text encoder ---
    logger.info("Exporting text encoder...")
    text_wrapper = OnnxTextEncoder(model)
    text_wrapper.eval()
    dummy_ids = torch.ones(1, 32, dtype=torch.long, device=device)
    dummy_mask = torch.ones(1, 32, dtype=torch.long, device=device)
    text_path = output_dir / f"{_TEXT_ENCODER_NAME}.onnx"
    torch.onnx.export(
        text_wrapper,
        (dummy_ids, dummy_mask),
        str(text_path),
        opset_version=opset_version,
        dynamo=False,
        input_names=["input_ids", "attention_mask"],
        output_names=["text_features", "text_mask"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
        },
    )
    exported[_TEXT_ENCODER_NAME] = text_path
    logger.info("  -> %s", text_path)

    # --- 3. Geometry encoder (classic) ---
    logger.info("Exporting geometry encoder (classic)...")
    geo_wrapper = OnnxGeometryEncoder(model, drop_spatial_bias=False)
    geo_wrapper.eval()
    # Compute FPN feature sizes from resolution
    # At 1x FPN scale: resolution / patch_size(14) = 72 (for 1008)
    feat_size = resolution // 14
    dummy_fpn = torch.randn(1, 256, feat_size, feat_size, device=device)
    dummy_pos = torch.randn(1, 256, feat_size, feat_size, device=device)
    dummy_boxes = torch.rand(1, 1, 4, device=device)
    dummy_box_labels = torch.ones(1, 1, dtype=torch.long, device=device)
    dummy_points = torch.rand(1, 1, 2, device=device)
    dummy_point_labels = torch.full((1, 1), -10, dtype=torch.long, device=device)

    geo_path = output_dir / f"{_GEOMETRY_ENCODER_NAME}.onnx"
    torch.onnx.export(
        geo_wrapper,
        (dummy_fpn, dummy_pos, dummy_boxes, dummy_box_labels, dummy_points, dummy_point_labels),
        str(geo_path),
        opset_version=opset_version,
        dynamo=False,
        input_names=[
            "fpn_feat_2",
            "fpn_pos_2",
            "input_boxes",
            "input_boxes_labels",
            "input_points",
            "input_points_labels",
        ],
        output_names=["geometry_features", "geometry_mask"],
        dynamic_axes={
            "input_boxes": {0: "batch", 1: "num_boxes"},
            "input_boxes_labels": {0: "batch", 1: "num_boxes"},
            "input_points": {0: "batch", 1: "num_points"},
            "input_points_labels": {0: "batch", 1: "num_points"},
        },
    )
    exported[_GEOMETRY_ENCODER_NAME] = geo_path
    logger.info("  -> %s", geo_path)

    # --- 4. Geometry encoder (exemplar — drop_spatial_bias=True) ---
    logger.info("Exporting geometry encoder (exemplar)...")
    geo_exemplar_wrapper = OnnxGeometryEncoder(model, drop_spatial_bias=True)
    geo_exemplar_wrapper.eval()
    # Exemplar mode uses points only (boxes converted to center points)
    dummy_boxes_ignore = torch.zeros(1, 1, 4, device=device)
    dummy_box_labels_ignore = torch.full((1, 1), -10, dtype=torch.long, device=device)
    dummy_ex_points = torch.rand(1, 1, 2, device=device)
    dummy_ex_point_labels = torch.ones(1, 1, dtype=torch.long, device=device)

    geo_exemplar_path = output_dir / f"{_GEOMETRY_ENCODER_EXEMPLAR_NAME}.onnx"
    torch.onnx.export(
        geo_exemplar_wrapper,
        (dummy_fpn, dummy_pos, dummy_boxes_ignore, dummy_box_labels_ignore, dummy_ex_points, dummy_ex_point_labels),
        str(geo_exemplar_path),
        opset_version=opset_version,
        dynamo=False,
        input_names=[
            "fpn_feat_2",
            "fpn_pos_2",
            "input_boxes",
            "input_boxes_labels",
            "input_points",
            "input_points_labels",
        ],
        output_names=["geometry_features", "geometry_mask"],
        dynamic_axes={
            "input_boxes": {0: "batch", 1: "num_boxes"},
            "input_boxes_labels": {0: "batch", 1: "num_boxes"},
            "input_points": {0: "batch", 1: "num_points"},
            "input_points_labels": {0: "batch", 1: "num_points"},
        },
    )
    exported[_GEOMETRY_ENCODER_EXEMPLAR_NAME] = geo_exemplar_path
    logger.info("  -> %s", geo_exemplar_path)

    # --- 5. Prompt decoder ---
    logger.info("Exporting prompt decoder...")
    decoder_wrapper = OnnxPromptDecoder(model)
    decoder_wrapper.eval()
    # FPN feature sizes: 4x -> resolution*4/14, 2x -> resolution*2/14, 1x -> resolution/14
    feat_4x = feat_size * 4  # 288 for 1008
    feat_2x = feat_size * 2  # 144 for 1008
    dummy_f0 = torch.randn(1, 256, feat_4x, feat_4x, device=device)
    dummy_f1 = torch.randn(1, 256, feat_2x, feat_2x, device=device)
    dummy_f2 = torch.randn(1, 256, feat_size, feat_size, device=device)
    dummy_p2 = torch.randn(1, 256, feat_size, feat_size, device=device)
    # Text-only prompt (32 tokens)
    dummy_prompt = torch.randn(1, 32, 256, device=device)
    dummy_pmask = torch.ones(1, 32, dtype=torch.bool, device=device)

    decoder_path = output_dir / f"{_PROMPT_DECODER_NAME}.onnx"
    torch.onnx.export(
        decoder_wrapper,
        (dummy_f0, dummy_f1, dummy_f2, dummy_p2, dummy_prompt, dummy_pmask),
        str(decoder_path),
        opset_version=opset_version,
        dynamo=False,
        input_names=[
            "fpn_feat_0",
            "fpn_feat_1",
            "fpn_feat_2",
            "fpn_pos_2",
            "prompt_features",
            "prompt_mask",
        ],
        output_names=["pred_masks", "pred_boxes", "pred_logits", "presence_logits"],
        dynamic_axes={
            "prompt_features": {0: "batch", 1: "prompt_len"},
            "prompt_mask": {0: "batch", 1: "prompt_len"},
        },
    )
    exported[_PROMPT_DECODER_NAME] = decoder_path
    logger.info("  -> %s", decoder_path)

    logger.info("ONNX export complete. %d models written to %s", len(exported), output_dir)
    return exported


def convert_onnx_to_openvino(
    onnx_dir: str | Path,
    output_dir: str | Path,
    *,
    compress_to_fp16: bool = True,
) -> dict[str, Path]:
    """Convert all SAM3 ONNX models in a directory to OpenVINO IR.

    Args:
        onnx_dir: Directory containing the ONNX files.
        output_dir: Directory to write OpenVINO IR files.
        compress_to_fp16: Compress weights to FP16 during conversion.

    Returns:
        Mapping from model name to the written ``.xml`` path.
    """
    import openvino as ov  # noqa: PLC0415

    onnx_dir = Path(onnx_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = [
        _VISION_ENCODER_NAME,
        _TEXT_ENCODER_NAME,
        _GEOMETRY_ENCODER_NAME,
        _GEOMETRY_ENCODER_EXEMPLAR_NAME,
        _PROMPT_DECODER_NAME,
    ]

    converted: dict[str, Path] = {}
    for name in model_names:
        onnx_path = onnx_dir / f"{name}.onnx"
        if not onnx_path.exists():
            logger.warning("Skipping %s — ONNX file not found.", onnx_path)
            continue

        logger.info("Converting %s to OpenVINO IR...", name)
        ov_model = ov.convert_model(str(onnx_path))
        ir_path = output_dir / f"{name}.xml"
        ov.save_model(ov_model, str(ir_path), compress_to_fp16=compress_to_fp16)
        converted[name] = ir_path
        logger.info("  -> %s", ir_path)

    logger.info("OpenVINO conversion complete. %d models written to %s", len(converted), output_dir)
    return converted
