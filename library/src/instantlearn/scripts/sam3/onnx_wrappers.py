# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ONNX-exportable ``nn.Module`` wrappers for SAM3 sub-components.

Each wrapper isolates a portion of ``Sam3Model`` into a standalone
``nn.Module`` with a clean ``forward()`` signature suitable for
``torch.onnx.export()``.

The 5-model split:

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

See Also:
    :mod:`instantlearn.scripts.sam3.export_sam3` — export pipeline and CLI
    that uses these wrappers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from instantlearn.models.sam3.model import Sam3Model


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
        from instantlearn.models.sam3.common import box_cxcywh_to_xyxy, inverse_sigmoid  # noqa: PLC0415

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
