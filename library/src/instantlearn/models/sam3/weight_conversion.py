# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Weight conversion utilities for loading sam3.pt into HuggingFace format.

The original Facebook checkpoint (sam3.pt) uses different parameter names and
layouts than our HuggingFace-style model.  This module handles:

1. Key remapping  (``SAM3_PT_TO_HF_KEY_MAPPING``)
2. QKV weight splitting  (combined ``in_proj`` → separate ``q/k/v_proj``)
3. Shape fixes  (position embeddings, text projection transpose)
"""

import re

import torch

# Key mapping from sam3.pt (original Facebook) to model.safetensors (HuggingFace) format
# fmt: off
SAM3_PT_TO_HF_KEY_MAPPING: dict[str, str] = {
    # Strip detector prefix
    r"^detector\.": r"",
    # Geometry encoder layer naming
    r"\.encode\.(\d+)\.": r".layers.\1.",
    r"\.cross_attn_image\.out_proj\.": r".cross_attn.o_proj.",
    r"\.cross_attn_image\.": r".cross_attn.",
    r"\.self_attn\.out_proj\.": r".self_attn.o_proj.",
    r"\.linear1\.": r".mlp.fc1.",
    r"\.linear2\.": r".mlp.fc2.",
    r"\.img_pre_norm\.": r".vision_layer_norm.",
    r"geometry_encoder\.norm\.": r"geometry_encoder.prompt_layer_norm.",
    r"\.encode_norm\.": r".output_layer_norm.",
    # Geometry encoder layer norms (norm1/2/3 -> layer_norm1/2/3)
    r"geometry_encoder\.layers\.(\d+)\.norm1\.": r"geometry_encoder.layers.\1.layer_norm1.",
    r"geometry_encoder\.layers\.(\d+)\.norm2\.": r"geometry_encoder.layers.\1.layer_norm2.",
    r"geometry_encoder\.layers\.(\d+)\.norm3\.": r"geometry_encoder.layers.\1.layer_norm3.",
    # DETR encoder layer naming
    r"^transformer\.encoder\.": r"detr_encoder.",
    # DETR encoder layer norms (norm1/2/3 -> layer_norm1/2/3)
    r"detr_encoder\.layers\.(\d+)\.norm1\.": r"detr_encoder.layers.\1.layer_norm1.",
    r"detr_encoder\.layers\.(\d+)\.norm2\.": r"detr_encoder.layers.\1.layer_norm2.",
    r"detr_encoder\.layers\.(\d+)\.norm3\.": r"detr_encoder.layers.\1.layer_norm3.",
    # DETR decoder layer naming
    r"^transformer\.decoder\.": r"detr_decoder.",
    r"\.ca_text\.out_proj\.": r".text_cross_attn.o_proj.",
    r"\.ca_text\.": r".text_cross_attn.",
    r"detr_decoder\.layers\.(\d+)\.cross_attn\.out_proj\.": r"detr_decoder.layers.\1.vision_cross_attn.o_proj.",
    r"detr_decoder\.layers\.(\d+)\.cross_attn\.": r"detr_decoder.layers.\1.vision_cross_attn.",
    r"detr_decoder\.layers\.(\d+)\.norm1\.": r"detr_decoder.layers.\1.vision_cross_attn_layer_norm.",
    r"detr_decoder\.layers\.(\d+)\.catext_norm\.": r"detr_decoder.layers.\1.text_cross_attn_layer_norm.",
    r"detr_decoder\.layers\.(\d+)\.norm2\.": r"detr_decoder.layers.\1.self_attn_layer_norm.",
    r"detr_decoder\.layers\.(\d+)\.norm3\.": r"detr_decoder.layers.\1.mlp_layer_norm.",
    r"\.bbox_embed\.layers\.0\.": r".box_head.layer1.",
    r"\.bbox_embed\.layers\.1\.": r".box_head.layer2.",
    r"\.bbox_embed\.layers\.2\.": r".box_head.layer3.",
    r"\.instance_bbox_embed\.layers\.0\.": r".instance_box_head.layer1.",
    r"\.instance_bbox_embed\.layers\.1\.": r".instance_box_head.layer2.",
    r"\.instance_bbox_embed\.layers\.2\.": r".instance_box_head.layer3.",
    r"\.ref_point_head\.layers\.0\.": r".ref_point_head.layer1.",
    r"\.ref_point_head\.layers\.1\.": r".ref_point_head.layer2.",
    r"\.boxRPB_embed_x\.layers\.0\.": r".box_rpb_embed_x.layer1.",
    r"\.boxRPB_embed_x\.layers\.1\.": r".box_rpb_embed_x.layer2.",
    r"\.boxRPB_embed_y\.layers\.0\.": r".box_rpb_embed_y.layer1.",
    r"\.boxRPB_embed_y\.layers\.1\.": r".box_rpb_embed_y.layer2.",
    r"detr_decoder\.norm\.": r"detr_decoder.output_layer_norm.",
    r"\.presence_token_head\.layers\.0\.": r".presence_head.layer1.",
    r"\.presence_token_head\.layers\.1\.": r".presence_head.layer2.",
    r"\.presence_token_head\.layers\.2\.": r".presence_head.layer3.",
    r"\.presence_token_out_norm\.": r".presence_layer_norm.",
    # Dot product scoring
    r"^dot_prod_scoring\.": r"dot_product_scoring.",
    r"\.prompt_mlp\.layers\.0\.": r".text_mlp.layer1.",
    r"\.prompt_mlp\.layers\.1\.": r".text_mlp.layer2.",
    r"\.prompt_mlp\.out_norm\.": r".text_mlp_out_norm.",
    r"\.prompt_proj\.": r".text_proj.",
    r"\.hs_proj\.": r".query_proj.",
    # Mask decoder
    r"^segmentation_head\.": r"mask_decoder.",
    r"\.pixel_decoder\.conv_layers\.": r".pixel_decoder.conv_layers.",
    r"\.pixel_decoder\.norms\.": r".pixel_decoder.norms.",
    r"\.mask_predictor\.mask_embed\.": r".mask_embedder.",
    r"\.mask_embed\.layers\.": r".mask_embedder.layers.",
    r"\.instance_seg_head\.": r".instance_projection.",
    r"\.semantic_seg_head\.": r".semantic_projection.",
    r"\.cross_attend_prompt\.out_proj\.": r".prompt_cross_attn.o_proj.",
    r"\.cross_attend_prompt\.": r".prompt_cross_attn.",
    r"\.cross_attn_norm\.": r".prompt_cross_attn_norm.",
    # Vision encoder
    r"^backbone\.vision_backbone\.trunk\.": r"vision_encoder.backbone.",
    r"vision_encoder\.backbone\.pos_embed": r"vision_encoder.backbone.embeddings.position_embeddings",
    r"vision_encoder\.backbone\.patch_embed\.proj\.": r"vision_encoder.backbone.embeddings.patch_embeddings.projection.",
    r"vision_encoder\.backbone\.ln_pre\.": r"vision_encoder.backbone.layer_norm.",
    r"vision_encoder\.backbone\.blocks\.(\d+)\.norm1\.": r"vision_encoder.backbone.layers.\1.layer_norm1.",
    r"vision_encoder\.backbone\.blocks\.(\d+)\.norm2\.": r"vision_encoder.backbone.layers.\1.layer_norm2.",
    r"vision_encoder\.backbone\.blocks\.(\d+)\.attn\.qkv\.": r"vision_encoder.backbone.layers.\1.attention.qkv.",
    r"vision_encoder\.backbone\.blocks\.(\d+)\.attn\.proj\.": r"vision_encoder.backbone.layers.\1.attention.o_proj.",
    r"vision_encoder\.backbone\.blocks\.(\d+)\.attn\.freqs_cis": r"vision_encoder.backbone.layers.\1.rotary_emb.rope_embeddings",
    r"vision_encoder\.backbone\.blocks\.(\d+)\.mlp\.fc1\.": r"vision_encoder.backbone.layers.\1.mlp.fc1.",
    r"vision_encoder\.backbone\.blocks\.(\d+)\.mlp\.fc2\.": r"vision_encoder.backbone.layers.\1.mlp.fc2.",
    # FPN
    r"^backbone\.vision_backbone\.neck\.fpn\.(\d+)\.": r"vision_encoder.neck.fpn_layers.\1.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2_0\.": r"vision_encoder.neck.fpn_layers.\1.scale_layers.0.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2_1\.": r"vision_encoder.neck.fpn_layers.\1.scale_layers.2.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2\.": r"vision_encoder.neck.fpn_layers.\1.scale_layers.0.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.maxpool_2x2\.": r"vision_encoder.neck.fpn_layers.\1.scale_layers.0.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.conv_1x1\.": r"vision_encoder.neck.fpn_layers.\1.proj1.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.conv_3x3\.": r"vision_encoder.neck.fpn_layers.\1.proj2.",
    # Text encoder
    r"^backbone\.language_backbone\.encoder\.": r"text_encoder.",
    r"^text_encoder\.token_embedding\.": r"text_encoder.text_model.embeddings.token_embedding.",
    r"^text_encoder\.positional_embedding": r"text_encoder.text_model.embeddings.position_embedding.weight",
    r"^text_encoder\.ln_final\.": r"text_encoder.text_model.final_layer_norm.",
    r"^text_encoder\.text_projection": r"text_encoder.text_projection.weight",
    r"text_encoder\.transformer\.resblocks\.(\d+)\.attn\.": r"text_encoder.text_model.encoder.layers.\1.self_attn.",
    r"text_encoder\.transformer\.resblocks\.(\d+)\.ln_1\.": r"text_encoder.text_model.encoder.layers.\1.layer_norm1.",
    r"text_encoder\.transformer\.resblocks\.(\d+)\.ln_2\.": r"text_encoder.text_model.encoder.layers.\1.layer_norm2.",
    r"text_encoder\.transformer\.resblocks\.(\d+)\.mlp\.c_fc\.": r"text_encoder.text_model.encoder.layers.\1.mlp.fc1.",
    r"text_encoder\.transformer\.resblocks\.(\d+)\.mlp\.c_proj\.": r"text_encoder.text_model.encoder.layers.\1.mlp.fc2.",
    r"^backbone\.language_backbone\.resizer\.": r"text_projection.",
}
# fmt: on


def _convert_sam3_pt_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert sam3.pt key names to HuggingFace format.

    Args:
        state_dict: State dict with original sam3.pt keys.

    Returns:
        State dict with converted keys matching HuggingFace format.
    """
    new_state_dict = {}
    for old_key, value in state_dict.items():
        new_key = old_key
        for pattern, replacement in SAM3_PT_TO_HF_KEY_MAPPING.items():
            new_key = re.sub(pattern, replacement, new_key)
        new_state_dict[new_key] = value
    return new_state_dict


def _split_qkv_weights(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Split combined QKV weights into separate Q, K, V projections.

    The original sam3.pt uses combined in_proj_weight [3*hidden, hidden] format.
    This function splits them into separate q_proj, k_proj, v_proj weights.

    Args:
        state_dict: State dict with combined QKV weights.

    Returns:
        State dict with split Q, K, V weights.
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        if "in_proj_weight" in key or "in_proj_bias" in key:
            # Split into 3 equal parts
            q, k, v = torch.chunk(value, 3, dim=0)

            if "in_proj_weight" in key:
                base_key = key.replace("in_proj_weight", "")
                new_state_dict[base_key + "q_proj.weight"] = q
                new_state_dict[base_key + "k_proj.weight"] = k
                new_state_dict[base_key + "v_proj.weight"] = v
            else:  # in_proj_bias
                base_key = key.replace("in_proj_bias", "")
                new_state_dict[base_key + "q_proj.bias"] = q
                new_state_dict[base_key + "k_proj.bias"] = k
                new_state_dict[base_key + "v_proj.bias"] = v
        elif ".qkv." in key:
            # Vision backbone uses .qkv. format
            q, k, v = torch.chunk(value, 3, dim=0)
            new_state_dict[key.replace(".qkv.", ".q_proj.")] = q
            new_state_dict[key.replace(".qkv.", ".k_proj.")] = k
            new_state_dict[key.replace(".qkv.", ".v_proj.")] = v
        else:
            new_state_dict[key] = value

    return new_state_dict


def _fix_shape_mismatches(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fix shape mismatches between sam3.pt and HuggingFace format.

    Handles:
    - Position embeddings: Strip CLS token position (577 -> 576)
    - Text projection: Transpose weight matrix

    Args:
        state_dict: State dict after key remapping.

    Returns:
        State dict with fixed shapes.
    """
    # Fix position embeddings: sam3.pt has [1, 577, 1024], HF expects [1, 576, 1024]
    # The first position is CLS token position which is not used
    pos_key = "vision_encoder.backbone.embeddings.position_embeddings"
    if pos_key in state_dict:
        pos_embed = state_dict[pos_key]
        if pos_embed.shape[1] == 577:
            state_dict[pos_key] = pos_embed[:, 1:, :]  # Strip CLS token position

    # Fix text projection: sam3.pt has [1024, 512], HF expects [512, 1024]
    text_proj_key = "text_encoder.text_projection.weight"
    if text_proj_key in state_dict:
        text_proj = state_dict[text_proj_key]
        if text_proj.shape == (1024, 512):
            state_dict[text_proj_key] = text_proj.t()  # Transpose

    return state_dict


def convert_sam3_pt_to_hf_format(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert sam3.pt state dict to HuggingFace-compatible format.

    Applies key remapping, QKV splitting, and shape fixes.

    Args:
        state_dict: Original sam3.pt state dict.

    Returns:
        Converted state dict compatible with HuggingFace model architecture.
    """
    state_dict = _convert_sam3_pt_keys(state_dict)
    state_dict = _split_qkv_weights(state_dict)
    return _fix_shape_mismatches(state_dict)
