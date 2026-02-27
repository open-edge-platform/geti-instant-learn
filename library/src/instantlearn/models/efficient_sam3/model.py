# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientSam3Model: student distilled variant of Sam3Model.

Inherits the full forward pipeline (DETR encoder/decoder, mask decoder,
geometry encoder, dot product scoring) from Sam3Model, replacing only the
vision encoder (timm backbone + projection head + FPN) and text encoder
(MobileCLIP-S1).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import timm
import torch
from huggingface_hub import hf_hub_download
from torch import nn

from instantlearn.models.sam3.model import Sam3Model

from .backbone import StudentVisionModel
from .constants import (
    BACKBONE_CONFIG,
    HF_REPO_ID,
    HF_SUBFOLDER,
    MOBILECLIP_S1_CONFIG,
    get_checkpoint_filename,
)
from .text_encoder import MobileCLIPTextEncoder

logger = logging.getLogger(__name__)


class EfficientSam3Model(Sam3Model):
    """EfficientSAM3 model with student backbone and MobileCLIP text encoder.

    Inherits the full detection and segmentation pipeline from Sam3Model.
    Overrides only the vision and text encoders for efficient inference while
    maintaining identical forward pass logic.

    Args:
        backbone_type: Backbone family ('efficientvit', 'repvit', 'tinyvit').
        variant: Model size variant (e.g. 'b0', 'm1_1', '11m').
        All remaining arguments are inherited from Sam3Model.
    """

    def __init__(
        self,
        backbone_type: str = "efficientvit",
        variant: str = "b2",
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize EfficientSam3Model.

        Args:
            backbone_type: Backbone family ('efficientvit', 'repvit', 'tinyvit').
            variant: Model size variant (e.g. 'b0', 'm1_1', '11m').
            **kwargs: Additional arguments passed to Sam3Model.__init__.
        """
        self._backbone_type = backbone_type
        self._variant = variant
        super().__init__(**kwargs)

        # Override parent's ViT vision encoder with student backbone + FPN
        self.vision_encoder = self._create_vision_encoder(**kwargs)

        # Override parent's CLIP text encoder with MobileCLIP-S1 and
        # re-create text_projection to match MobileCLIP hidden dim (512)
        # instead of CLIP hidden dim (1024)
        self.text_encoder, self.text_projection = self._create_text_encoder(**kwargs)

    def _create_vision_encoder(self, **kwargs: Any) -> nn.Module:  # noqa: ANN401
        """Create student vision encoder with timm backbone + FPN.

        Ignores ViT-specific parameters from parent and uses timm backbone instead.

        Returns:
            StudentVisionModel producing SAM3-compatible output dict.
        """
        fpn_hidden_size = kwargs.get("fpn_hidden_size", 256)
        return StudentVisionModel(
            backbone_type=self._backbone_type,
            variant=self._variant,
            fpn_hidden_size=fpn_hidden_size,
        )

    def _create_text_encoder(self, **kwargs: Any) -> tuple[nn.Module, nn.Module]:  # noqa: ANN401, PLR6301
        """Create MobileCLIP-S1 text encoder and linear projection.

        Ignores CLIP-specific parameters from parent and uses MobileCLIP-S1.

        Returns:
            (MobileCLIPTextEncoder, nn.Linear) tuple for text encoding + projection.
        """
        detr_hidden = kwargs.get("detr_encoder_hidden_size", 256)
        config = MOBILECLIP_S1_CONFIG
        encoder = MobileCLIPTextEncoder(
            vocab_size=config["vocab_size"],
            dim=config["dim"],
            n_layers=config["n_transformer_layers"],
            n_heads=config["n_heads_per_layer"],
            ffn_multiplier=config["ffn_multiplier_per_layer"],
            context_length=config["context_length"],
        )
        projection = nn.Linear(config["dim"], detr_hidden)
        return encoder, projection

    def get_text_features(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> SimpleNamespace:
        """Extract text features from MobileCLIP text encoder.

        Returns a namespace with ``pooler_output`` (projected features) that is
        compatible with the parent class's forward method.

        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len]. Default: None.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            SimpleNamespace with pooler_output [batch_size, seq_len, detr_hidden]
            and last_hidden_state [batch_size, seq_len, dim].
        """
        all_tokens, _ = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        projected = self.text_projection(all_tokens)
        return SimpleNamespace(
            pooler_output=projected,
            last_hidden_state=all_tokens,
        )

    def _get_scoring_features(  # noqa: PLR6301
        self,
        text_features: torch.Tensor,
        text_mask: torch.Tensor | None,
        encoder_text_features: torch.Tensor,  # noqa: ARG002
        combined_prompt_mask: torch.Tensor | None,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Use text-only features for scoring in EfficientSAM3.

        Geometry embeddings (boxes/points) influence the DETR encoder/decoder
        via cross-attention, but including them in ``DotProductScoring``'s
        mean-pooled features dilutes the text signal and degrades scoring
        quality — particularly for multi-point prompts. The distilled student
        was trained with text-only scoring, so we preserve that here.

        Args:
            text_features: Raw text encoder output
                [batch_size, seq_len, hidden_size].
            text_mask: Mask for raw text features [batch_size, seq_len].
            encoder_text_features: Unused (DETR encoder combined features).
            combined_prompt_mask: Unused (combined mask).

        Returns:
            (text_features, text_mask) for text-only scoring.
        """
        return text_features, text_mask

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | None = None,
        backbone_type: str = "efficientvit",
        variant: str = "b2",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        torch_dtype: torch.dtype | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> EfficientSam3Model:
        """Load a pretrained EfficientSAM3 model from HuggingFace Hub or local path.

        Args:
            pretrained_model_name_or_path: Local .pth path, or None to auto-download
                from HuggingFace based on backbone_type/variant.
            backbone_type: Backbone family.
            variant: Model size variant.
            device: Target device.
            dtype: Data type for model weights (alias for torch_dtype).
            torch_dtype: Data type for model weights.
            **kwargs: Additional arguments passed to EfficientSam3Model.__init__.

        Returns:
            Loaded EfficientSam3Model instance.
        """
        if torch_dtype is not None and dtype is None:
            dtype = torch_dtype

        # Resolve checkpoint path
        if pretrained_model_name_or_path is not None:
            path = Path(pretrained_model_name_or_path)
            if path.exists():
                model_path = str(path)
            else:
                model_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename=get_checkpoint_filename(backbone_type, variant),
                    subfolder=HF_SUBFOLDER,
                )
        else:
            filename = get_checkpoint_filename(backbone_type, variant)
            model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                subfolder=HF_SUBFOLDER,
            )

        logger.info("Loading EfficientSAM3 checkpoint: %s", model_path)
        raw_state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        # Extract from nested 'model' key if present (HuggingFace checkpoint format)
        if "model" in raw_state_dict and isinstance(raw_state_dict["model"], dict):
            raw_state_dict = raw_state_dict["model"]

        # Convert key naming from original SAM3 repo to our model format
        state_dict = convert_efficientsam3_state_dict(
            raw_state_dict,
            backbone_type=backbone_type,
            variant=variant,
        )

        # Create model
        model = cls(backbone_type=backbone_type, variant=variant, **kwargs)

        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        # Disable the presence_token mechanism for EfficientSAM3.
        #
        # The presence_token_head doesn't work well for EfficientSAM3 due to
        # the model being a distilled student. It produces unreliable presence
        # scores that are often too low for simple text prompts, making the
        # mask probabilities very low. See:
        # https://github.com/SimonZeng7108/efficientsam3/issues/17
        model.detr_decoder.use_presence = False
        logger.info(
            "Presence token disabled for EfficientSAM3 — scores use "
            "pred_logits only (presence head unreliable for distilled model).",
        )

        # Filter presence keys from missing list (intentionally unused)
        presence_prefixes = (
            "detr_decoder.presence_token.",
            "detr_decoder.presence_head.",
            "detr_decoder.presence_layer_norm.",
        )
        missing_keys = [k for k in missing_keys if not any(k.startswith(p) for p in presence_prefixes)]

        if missing_keys:
            logger.warning("Missing keys when loading EfficientSAM3: %s", missing_keys)
        if unexpected_keys:
            logger.warning("Unexpected keys when loading EfficientSAM3: %s", unexpected_keys)

        if device is not None:
            model = model.to(device)
        if dtype is not None:
            model = model.to(dtype)

        return model


# ---------------------------------------------------------------------------
# State dict conversion: original SAM3 repo format → our model format
# ---------------------------------------------------------------------------

# Keys with these prefixes are skipped entirely during conversion.
_SKIP_PREFIXES: tuple[str, ...] = (
    # SAM2 video tracker (not used in image-only EfficientSAM3)
    "detector.tracker.",
    "tracker.",
    # SAM2 FPN convs (separate from EfficientSAM3 FPN)
    "detector.backbone.vision_backbone.sam2_convs.",
    # Teacher ViT encoder (only student backbone is needed)
    "detector.backbone.vision_backbone.trunk.blocks.",
    "detector.backbone.vision_backbone.trunk.patch_embed.",
    "detector.backbone.vision_backbone.trunk.ln_pre.",
)

_SKIP_EXACT: frozenset[str] = frozenset({
    "detector.backbone.vision_backbone.trunk.pos_embed",
})

# Ordered list of (regex_pattern, replacement) for key renaming.
# Applied sequentially in this order.
_KEY_RENAME_RULES: list[tuple[str, str]] = [
    # ---- Top-level prefix renaming ----
    # 1. Strip "detector." prefix
    (r"^detector\.", ""),
    # 2. Vision backbone → vision_encoder
    (r"^backbone\.vision_backbone\.", "vision_encoder."),
    # 3. Language backbone encoder → text_encoder
    (r"^backbone\.language_backbone\.encoder\.", "text_encoder."),
    # 4. Language backbone projector → text_projection
    (r"^backbone\.language_backbone\.projector\.", "text_projection."),
    # 5. DETR transformer → detr_encoder / detr_decoder
    (r"^transformer\.encoder\.", "detr_encoder."),
    (r"^transformer\.decoder\.", "detr_decoder."),
    # 6. Segmentation head → mask_decoder
    (r"^segmentation_head\.", "mask_decoder."),
    # 7. Dot product scoring
    (r"^dot_prod_scoring\.", "dot_product_scoring."),
    # ---- DETR Encoder layer renaming ----
    (r"(detr_encoder\.layers\.\d+\.)cross_attn_image\.", r"\1cross_attn."),
    (r"(detr_encoder\.layers\.\d+\.)norm(\d)\.", r"\1layer_norm\2."),
    (r"(detr_encoder\.layers\.\d+\.)linear1\.", r"\1mlp.fc1."),
    (r"(detr_encoder\.layers\.\d+\.)linear2\.", r"\1mlp.fc2."),
    # ---- DETR Decoder layer renaming ----
    # NOTE: cross_attn before norm to avoid partial matches
    (r"(detr_decoder\.layers\.\d+\.)cross_attn\.", r"\1vision_cross_attn."),
    (r"(detr_decoder\.layers\.\d+\.)ca_text\.", r"\1text_cross_attn."),
    (r"(detr_decoder\.layers\.\d+\.)catext_norm\.", r"\1text_cross_attn_layer_norm."),
    (r"(detr_decoder\.layers\.\d+\.)norm1\.", r"\1vision_cross_attn_layer_norm."),
    (r"(detr_decoder\.layers\.\d+\.)norm2\.", r"\1self_attn_layer_norm."),
    (r"(detr_decoder\.layers\.\d+\.)norm3\.", r"\1mlp_layer_norm."),
    (r"(detr_decoder\.layers\.\d+\.)linear1\.", r"\1mlp.fc1."),
    (r"(detr_decoder\.layers\.\d+\.)linear2\.", r"\1mlp.fc2."),
    # ---- Decoder top-level renaming ----
    (r"^detr_decoder\.norm\.", "detr_decoder.output_layer_norm."),
    (r"^detr_decoder\.bbox_embed\.", "detr_decoder.box_head."),
    (r"^detr_decoder\.presence_token_head\.", "detr_decoder.presence_head."),
    (r"^detr_decoder\.presence_token_out_norm\.", "detr_decoder.presence_layer_norm."),
    (r"^detr_decoder\.boxRPB_embed_x\.", "detr_decoder.box_rpb_embed_x."),
    (r"^detr_decoder\.boxRPB_embed_y\.", "detr_decoder.box_rpb_embed_y."),
    # ---- Geometry encoder renaming ----
    (r"^geometry_encoder\.encode\.", "geometry_encoder.layers."),
    (r"^geometry_encoder\.encode_norm\.", "geometry_encoder.output_layer_norm."),
    (r"^geometry_encoder\.img_pre_norm\.", "geometry_encoder.vision_layer_norm."),
    (r"^geometry_encoder\.norm\.", "geometry_encoder.prompt_layer_norm."),
    (r"(geometry_encoder\.layers\.\d+\.)cross_attn_image\.", r"\1cross_attn."),
    (r"(geometry_encoder\.layers\.\d+\.)norm(\d)\.", r"\1layer_norm\2."),
    (r"(geometry_encoder\.layers\.\d+\.)linear1\.", r"\1mlp.fc1."),
    (r"(geometry_encoder\.layers\.\d+\.)linear2\.", r"\1mlp.fc2."),
    # ---- DotProductScoring renaming ----
    (r"^dot_product_scoring\.prompt_proj\.", "dot_product_scoring.text_proj."),
    (r"^dot_product_scoring\.hs_proj\.", "dot_product_scoring.query_proj."),
    (r"^dot_product_scoring\.prompt_mlp\.", "dot_product_scoring.text_mlp."),
    # text_mlp.out_norm → text_mlp_out_norm (sibling attribute, not nested)
    (r"^dot_product_scoring\.text_mlp\.out_norm\.", "dot_product_scoring.text_mlp_out_norm."),
    # ---- MaskDecoder renaming ----
    (r"^mask_decoder\.mask_predictor\.mask_embed\.", "mask_decoder.mask_embedder."),
    (r"^mask_decoder\.cross_attend_prompt\.", "mask_decoder.prompt_cross_attn."),
    (r"^mask_decoder\.cross_attn_norm\.", "mask_decoder.prompt_cross_attn_norm."),
    (r"^mask_decoder\.instance_seg_head\.", "mask_decoder.instance_projection."),
    (r"^mask_decoder\.semantic_seg_head\.", "mask_decoder.semantic_projection."),
    # ---- Vision encoder: student backbone ----
    # Projection head: trunk.model.head → trunk.projector.head
    (r"^vision_encoder\.trunk\.model\.head\.", "vision_encoder.trunk.projector.head."),
    # Student backbone: trunk.model.backbone.model → trunk.student_trunk.model
    (
        r"^vision_encoder\.trunk\.model\.backbone\.model\.",
        "vision_encoder.trunk.student_trunk.model.",
    ),
    # Stem: input_stem.op_list.0 → stem_in_conv, input_stem.op_list.1 → stem_res0
    (
        r"(trunk\.student_trunk\.model\.)input_stem\.op_list\.0\.",
        r"\1stem_in_conv.",
    ),
    (
        r"(trunk\.student_trunk\.model\.)input_stem\.op_list\.1\.",
        r"\1stem_res0.",
    ),
    # Stages: stages.N.op_list.M → stages_N.blocks.M
    (
        r"(trunk\.student_trunk\.model\.)stages\.(\d+)\.op_list\.(\d+)\.",
        r"\1stages_\2.blocks.\3.",
    ),
    # ---- Text encoder: Sequential → named attributes ----
    # pre_norm_mha.0 (LayerNorm) → pre_norm_mha
    (r"(text_encoder\.transformer\.\d+\.)pre_norm_mha\.0\.", r"\1pre_norm_mha."),
    # pre_norm_mha.1 (MultiHeadSelfAttention) → self_attn
    (r"(text_encoder\.transformer\.\d+\.)pre_norm_mha\.1\.", r"\1self_attn."),
    # pre_norm_ffn.0 (LayerNorm) → pre_norm_ffn
    (r"(text_encoder\.transformer\.\d+\.)pre_norm_ffn\.0\.", r"\1pre_norm_ffn."),
    # pre_norm_ffn.1 (Linear in) → ffn.0
    (r"(text_encoder\.transformer\.\d+\.)pre_norm_ffn\.1\.", r"\1ffn.0."),
    # pre_norm_ffn.4 (Linear out) → ffn.2  (indices 2,3 are GELU+Dropout, no params)
    (r"(text_encoder\.transformer\.\d+\.)pre_norm_ffn\.4\.", r"\1ffn.2."),
    # ---- Text encoder: positional embedding dedup ----
    (r"positional_embedding\.pos_embed\.pos_embed", "positional_embedding.pos_embed"),
]

# Regex matching DecoderMLP instances that use .layers.N naming in checkpoint.
# These modules use nn.ModuleList in the original code but named attributes
# (layer1, layer2, layer3) in our DecoderMLP implementation.
_DECODER_MLP_PATTERN: re.Pattern[str] = re.compile(
    r"^("
    r"detr_decoder\.(?:box_head|ref_point_head|presence_head|box_rpb_embed_[xy])"
    r"|dot_product_scoring\.text_mlp"
    r")\.layers\.(\d+)\.",
)


def _apply_key_renames(key: str) -> str:
    """Apply all regex renaming rules to a single state dict key.

    Args:
        key: Original state dict key.

    Returns:
        Transformed key matching our model's naming convention.
    """
    for pattern, replacement in _KEY_RENAME_RULES:
        key = re.sub(pattern, replacement, key)
    return key


def _convert_decoder_mlp_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert DecoderMLP .layers.N keys to .layerN+1 naming.

    The original SAM3 code uses nn.ModuleList for MLP layers, producing keys
    like ``box_head.layers.0.weight``. Our DecoderMLP uses named attributes
    ``layer1``, ``layer2``, ``layer3`` instead.

    Args:
        state_dict: State dict after regex renaming.

    Returns:
        State dict with DecoderMLP keys converted.
    """
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        match = _DECODER_MLP_PATTERN.match(key)
        if match:
            prefix = match.group(1)
            idx = int(match.group(2))
            suffix = key[match.end() :]
            new_key = f"{prefix}.layer{idx + 1}.{suffix}"
            converted[new_key] = value
        else:
            converted[key] = value
    return converted


def _split_fused_qkv(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Split fused nn.MultiheadAttention QKV weights into separate projections.

    Handles the conversion from ``in_proj_weight``/``in_proj_bias`` (fused format)
    to separate ``q_proj``/``k_proj``/``v_proj`` weights matching our Attention class.
    Also renames ``out_proj`` to ``o_proj``.

    Only applies to attention modules under detr_encoder, detr_decoder,
    geometry_encoder, and mask_decoder prefixes.

    Args:
        state_dict: State dict with renamed keys (after _apply_key_renames).

    Returns:
        State dict with split QKV projections and renamed output projections.
    """
    converted: dict[str, torch.Tensor] = {}
    attn_prefixes = ("detr_encoder.", "detr_decoder.", "geometry_encoder.", "mask_decoder.")

    for key, value in state_dict.items():
        is_attn_scope = any(key.startswith(p) for p in attn_prefixes)

        if is_attn_scope and "in_proj_weight" in key:
            dim = value.shape[0] // 3
            base = key.replace(".in_proj_weight", "")
            converted[f"{base}.q_proj.weight"] = value[:dim]
            converted[f"{base}.k_proj.weight"] = value[dim : 2 * dim]
            converted[f"{base}.v_proj.weight"] = value[2 * dim :]
        elif is_attn_scope and "in_proj_bias" in key:
            dim = value.shape[0] // 3
            base = key.replace(".in_proj_bias", "")
            converted[f"{base}.q_proj.bias"] = value[:dim]
            converted[f"{base}.k_proj.bias"] = value[dim : 2 * dim]
            converted[f"{base}.v_proj.bias"] = value[2 * dim :]
        elif is_attn_scope and ".out_proj." in key:
            # Rename out_proj → o_proj for our Attention class
            new_key = key.replace(".out_proj.", ".o_proj.")
            converted[new_key] = value
        else:
            converted[key] = value

    return converted


def convert_efficientsam3_state_dict(
    raw_state_dict: dict[str, torch.Tensor],
    backbone_type: str | None = None,
    variant: str | None = None,
) -> dict[str, torch.Tensor]:
    """Convert an EfficientSAM3 checkpoint state dict to our model format.

    Handles the following conversions:
    1. Filtering of teacher ViT, SAM2 tracker, and sam2_convs keys
    2. Module prefix renaming (detector → our hierarchy)
    3. Layer attribute renaming (norm → layer_norm, etc.)
    4. Vision encoder student backbone key mapping (checkpoint → timm naming)
    5. Backbone-specific key conversion (RepViT flat→hierarchical)
    6. Text encoder Sequential numbering → named attributes
    7. Positional embedding deduplication
    8. DecoderMLP .layers.N → .layerN+1 conversion
    9. DotProductScoring text_mlp.out_norm → text_mlp_out_norm
    10. Fused QKV weight splitting (nn.MultiheadAttention → separate Q/K/V)

    Note: TinyViT backbone keys need no conversion because segment_anything_hq
    overrides timm's TinyViT model registrations, so the model uses the same
    naming as the original checkpoint (layers.N, .c., patch_embed.seq).

    Args:
        raw_state_dict: Raw checkpoint state dict (with "detector." prefix).
        backbone_type: Backbone family ('efficientvit', 'repvit', 'tinyvit').
            When provided, applies backbone-specific key conversion for RepViT.
        variant: Model size variant (e.g. 'b0', 'm1_1', '11m').
            Required when backbone_type is 'repvit'.

    Returns:
        Converted state dict compatible with EfficientSam3Model.load_state_dict().
    """
    # Step 1: Filter out keys we don't need (teacher ViT, tracker, sam2_convs)
    filtered: dict[str, torch.Tensor] = {}
    for key, value in raw_state_dict.items():
        if key in _SKIP_EXACT:
            continue
        if any(key.startswith(p) for p in _SKIP_PREFIXES):
            continue
        filtered[key] = value

    logger.debug(
        "State dict: %d keys total, %d after filtering",
        len(raw_state_dict),
        len(filtered),
    )

    # Step 2: Apply all regex key renames
    renamed: dict[str, torch.Tensor] = {}
    for key, value in filtered.items():
        renamed[_apply_key_renames(key)] = value

    # Step 3: Backbone-specific key conversion
    if backbone_type == "repvit" and variant is not None:
        timm_name = BACKBONE_CONFIG[backbone_type, variant][0]
        renamed = _convert_repvit_backbone_keys(renamed, timm_name)

    # Step 4: Convert DecoderMLP .layers.N → .layerN+1
    renamed = _convert_decoder_mlp_keys(renamed)

    # Step 5: Split fused QKV and rename out_proj → o_proj
    return _split_fused_qkv(renamed)


# ---------------------------------------------------------------------------
# Backbone-specific key converters
# ---------------------------------------------------------------------------

# Note: TinyViT backbone keys do NOT require conversion. The
# segment_anything_hq package overrides timm's TinyViT model registrations,
# so the timm-created backbone uses the same naming as the original SAM3
# checkpoint (layers.N, patch_embed.seq, .c.). The SAM-HQ TinyViT also
# retains its classifier head/norm_head parameters despite features_only=True;
# these are unused at inference but load from the checkpoint without warnings.

_BACKBONE_PREFIX = "vision_encoder.trunk.student_trunk.model."


def _convert_repvit_backbone_keys(
    state_dict: dict[str, torch.Tensor],
    timm_name: str,
) -> dict[str, torch.Tensor]:
    """Convert RepViT backbone keys from flat features list to timm hierarchy.

    The original RepViT uses a flat ``nn.ModuleList`` called ``features``
    where features[0] is the stem and features[1..N] are ``RepViTBlock``
    instances. The timm implementation reorganizes blocks into hierarchical
    stages with explicit downsample modules.

    This function builds the flat→hierarchical mapping by inspecting the
    timm model structure, then applies per-block key renaming.

    Args:
        state_dict: State dict after generic regex renaming.
        timm_name: timm model name (e.g. 'repvit_m0_9.dist_450e_in1k').
    """
    block_mapping = _build_repvit_block_mapping(timm_name)

    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not key.startswith(_BACKBONE_PREFIX):
            converted[key] = value
            continue

        suffix = key[len(_BACKBONE_PREFIX) :]

        # Skip classifier head keys
        if suffix.startswith(("classifier.", "head.")):
            continue

        new_suffix = _rename_repvit_suffix(suffix, block_mapping)
        converted[_BACKBONE_PREFIX + new_suffix] = value

    return converted


def _build_repvit_block_mapping(
    timm_name: str,
) -> list[tuple[int, str, bool]]:
    """Build ordered mapping from flat feature indices to timm hierarchy.

    Returns a list of (flat_index, timm_prefix, is_stride2) tuples describing
    how each RepViTBlock in the original flat features list maps to the
    timm model's hierarchical stage/block/downsample structure.

    Args:
        timm_name: timm model name for the RepViT variant.

    Returns:
        List of (flat_idx, timm_path, is_stride2_block) tuples.
    """
    model = timm.create_model(
        timm_name,
        pretrained=False,
        features_only=True,
        out_indices=(-1,),
    )

    mapping: list[tuple[int, str, bool]] = []
    flat_idx = 1  # features[0] is the stem, blocks start at 1

    for stage_idx in range(len([n for n in dir(model) if n.startswith("stages_")])):
        stage = getattr(model, f"stages_{stage_idx}", None)
        if stage is None:
            break

        # Downsample: pre_block (regular) + stride-2 block (split)
        has_downsample = hasattr(stage, "downsample") and not isinstance(stage.downsample, nn.Identity)
        if has_downsample:
            mapping.append(
                (flat_idx, f"stages_{stage_idx}.downsample.pre_block", False),
            )
            flat_idx += 1
            mapping.append(
                (flat_idx, f"stages_{stage_idx}.downsample", True),
            )
            flat_idx += 1

        # Regular blocks
        for block_idx in range(len(stage.blocks)):
            mapping.append(
                (flat_idx, f"stages_{stage_idx}.blocks.{block_idx}", False),
            )
            flat_idx += 1

    return mapping


def _rename_repvit_suffix(
    suffix: str,
    block_mapping: list[tuple[int, str, bool]],
) -> str:
    """Rename a single RepViT key suffix from original to timm format.

    Args:
        suffix: Key suffix after removing the backbone prefix.
        block_mapping: Flat→hierarchical mapping from _build_repvit_block_mapping.

    Returns:
        Renamed suffix matching timm's naming convention.
    """
    # Stem: features.0.{0,2} → stem.{conv1,conv2}
    if suffix.startswith("features.0.0."):
        return "stem.conv1." + suffix[len("features.0.0.") :]
    if suffix.startswith("features.0.2."):
        return "stem.conv2." + suffix[len("features.0.2.") :]

    # Find matching block by flat index
    for flat_idx, timm_prefix, is_stride2 in block_mapping:
        feat_prefix = f"features.{flat_idx}."
        if not suffix.startswith(feat_prefix):
            continue

        remainder = suffix[len(feat_prefix) :]
        if is_stride2:
            return _rename_repvit_stride2_block(timm_prefix, remainder)
        return _rename_repvit_regular_block(timm_prefix, remainder)

    return suffix  # Pass through unmatched keys


def _rename_repvit_regular_block(prefix: str, remainder: str) -> str:
    """Rename internals of a regular RepViT block.

    Original Sequential-wrapped token_mixer and Residual-wrapped channel_mixer
    are flattened to direct attributes in timm.

    Mappings:
    - ``token_mixer.0.X`` → ``token_mixer.X`` (strip Sequential index)
    - ``token_mixer.1.X`` → ``se.X`` (SqueezeExcite moved to block level)
    - ``channel_mixer.m.0.X`` → ``channel_mixer.conv1.X``
    - ``channel_mixer.m.2.X`` → ``channel_mixer.conv2.X`` (index 1 is GELU)
    """
    rules: list[tuple[str, str]] = [
        ("token_mixer.0.", "token_mixer."),
        ("token_mixer.1.", "se."),
        ("channel_mixer.m.0.", "channel_mixer.conv1."),
        ("channel_mixer.m.2.", "channel_mixer.conv2."),
    ]
    for old, new in rules:
        if remainder.startswith(old):
            return f"{prefix}.{new}{remainder[len(old) :]}"
    return f"{prefix}.{remainder}"


def _rename_repvit_stride2_block(prefix: str, remainder: str) -> str:
    """Rename internals of a stride-2 RepViT block into downsample components.

    The original stride-2 RepViTBlock is split into separate downsample
    sub-modules in timm:

    Mappings:
    - ``token_mixer.0.X`` → ``spatial_downsample.X`` (depthwise stride-2 conv)
    - ``token_mixer.2.X`` → ``channel_downsample.X`` (1x1 channel projection)
    - ``channel_mixer.m.0.X`` → ``ffn.conv1.X``
    - ``channel_mixer.m.2.X`` → ``ffn.conv2.X``
    """
    rules: list[tuple[str, str]] = [
        ("token_mixer.0.", "spatial_downsample."),
        ("token_mixer.2.", "channel_downsample."),
        ("channel_mixer.m.0.", "ffn.conv1."),
        ("channel_mixer.m.2.", "ffn.conv2."),
    ]
    for old, new in rules:
        if remainder.startswith(old):
            return f"{prefix}.{new}{remainder[len(old) :]}"
    return f"{prefix}.{remainder}"
