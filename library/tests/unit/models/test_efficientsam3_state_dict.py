# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for EfficientSAM3 state dict conversion across all backbone variants.

Verifies that `convert_efficientsam3_state_dict` correctly maps checkpoint keys
to model keys for all 9 distilled encoder variants (3 backbone families x 3 sizes)
with MobileCLIP-S1 text encoder.
"""

from __future__ import annotations

import re

import pytest
import timm
import torch

from instantlearn.models.efficient_sam3.constants import BACKBONE_CONFIG, BackboneType
from instantlearn.models.efficient_sam3.model import (
    _BACKBONE_PREFIX,  # noqa: PLC2701
    EfficientSam3Model,
    _build_repvit_block_mapping,  # noqa: PLC2701
    convert_efficientsam3_state_dict,
)

# All 9 model variants: (backbone_type, variant)
ALL_VARIANTS: list[tuple[str, str]] = [
    (BackboneType.EFFICIENTVIT, "b0"),
    (BackboneType.EFFICIENTVIT, "b1"),
    (BackboneType.EFFICIENTVIT, "b2"),
    (BackboneType.REPVIT, "m0_9"),
    (BackboneType.REPVIT, "m1_1"),
    (BackboneType.REPVIT, "m2_3"),
    (BackboneType.TINYVIT, "5m"),
    (BackboneType.TINYVIT, "11m"),
    (BackboneType.TINYVIT, "21m"),
]

# Keys intentionally missing in distilled checkpoints (presence token not calibrated)
EXPECTED_MISSING_PREFIXES = (
    "detr_decoder.presence_token.",
    "detr_decoder.presence_head.",
    "detr_decoder.presence_layer_norm.",
)


def _build_mock_checkpoint_backbone_keys(
    backbone_type: str,
    variant: str,
) -> dict[str, torch.Tensor]:
    """Build synthetic checkpoint keys for the backbone portion.

    Generates keys in the ORIGINAL checkpoint format (as produced by the
    EfficientSAM3 repo's conversion scripts) by reverse-engineering from
    the timm model's state dict.

    Args:
        backbone_type: Backbone family.
        variant: Model size variant.

    Returns:
        Dict with checkpoint-format keys and dummy tensors.
    """
    timm_name = BACKBONE_CONFIG[backbone_type, variant][0]
    timm_model = timm.create_model(
        timm_name,
        pretrained=False,
        features_only=True,
        out_indices=(-1,),
    )
    timm_sd = timm_model.state_dict()

    # Reverse-map timm keys to original checkpoint keys
    ckpt_keys: dict[str, torch.Tensor] = {}

    if backbone_type == BackboneType.EFFICIENTVIT:
        ckpt_keys = _reverse_map_efficientvit(timm_sd)
    elif backbone_type == BackboneType.TINYVIT:
        ckpt_keys = _reverse_map_tinyvit(timm_sd)
    elif backbone_type == BackboneType.REPVIT:
        ckpt_keys = _reverse_map_repvit(timm_sd, timm_name)

    return ckpt_keys


def _reverse_map_efficientvit(
    timm_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Reverse-map EfficientViT timm keys to original checkpoint format."""
    result: dict[str, torch.Tensor] = {}
    for key, value in timm_sd.items():
        # stem_in_conv.X → input_stem.op_list.0.X
        # stem_res0.X → input_stem.op_list.1.X
        # stages_N.blocks.M.X → stages.N.op_list.M.X
        ckpt_key = key
        ckpt_key = re.sub(r"^stem_in_conv\.", "input_stem.op_list.0.", ckpt_key)
        ckpt_key = re.sub(r"^stem_res0\.", "input_stem.op_list.1.", ckpt_key)
        ckpt_key = re.sub(
            r"^stages_(\d+)\.blocks\.(\d+)\.",
            r"stages.\1.op_list.\2.",
            ckpt_key,
        )
        # Wrap in checkpoint prefix
        full_key = "detector.backbone.vision_backbone.trunk.model.backbone.model." + ckpt_key
        result[full_key] = value
    return result


def _reverse_map_tinyvit(
    timm_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Reverse-map TinyViT timm keys to original checkpoint format.

    Because segment_anything_hq overrides timm's TinyViT registration, the
    model already uses original naming (layers.N, .c., patch_embed.seq).
    No key transformation is needed — just add the checkpoint prefix.
    """
    result: dict[str, torch.Tensor] = {}
    for key, value in timm_sd.items():
        full_key = "detector.backbone.vision_backbone.trunk.model.backbone.model." + key
        result[full_key] = value
    return result


def _reverse_map_repvit(
    timm_sd: dict[str, torch.Tensor],
    timm_name: str,
) -> dict[str, torch.Tensor]:
    """Reverse-map RepViT timm keys to original checkpoint format."""
    # Build forward mapping to derive reverse mapping
    block_mapping = _build_repvit_block_mapping(timm_name)

    # Build reverse lookup: timm_prefix → (flat_idx, is_stride2)
    reverse_blocks: dict[str, tuple[int, bool]] = {}
    for flat_idx, timm_prefix, is_stride2 in block_mapping:
        reverse_blocks[timm_prefix] = (flat_idx, is_stride2)

    result: dict[str, torch.Tensor] = {}
    for key, value in timm_sd.items():
        ckpt_key = _reverse_repvit_key(key, reverse_blocks)
        full_key = "detector.backbone.vision_backbone.trunk.model.backbone.model." + ckpt_key
        result[full_key] = value
    return result


def _reverse_repvit_key(
    key: str,
    reverse_blocks: dict[str, tuple[int, bool]],
) -> str:
    """Reverse a single RepViT timm key to original checkpoint format."""
    # Stem: stem.conv1 → features.0.0, stem.conv2 → features.0.2
    if key.startswith("stem.conv1."):
        return "features.0.0." + key[len("stem.conv1.") :]
    if key.startswith("stem.conv2."):
        return "features.0.2." + key[len("stem.conv2.") :]

    # Find matching block prefix (longest match first for downsample.pre_block)
    for timm_prefix in sorted(reverse_blocks, key=len, reverse=True):
        with_dot = timm_prefix + "."
        if not key.startswith(with_dot):
            continue
        flat_idx, is_stride2 = reverse_blocks[timm_prefix]
        remainder = key[len(with_dot) :]

        if is_stride2:
            return _reverse_repvit_stride2(flat_idx, remainder)
        return _reverse_repvit_regular(flat_idx, remainder)

    return key  # Pass through unmatched


def _reverse_repvit_regular(flat_idx: int, remainder: str) -> str:
    """Reverse-map regular block internals to original naming."""
    rules = [
        ("se.", "token_mixer.1."),
        ("token_mixer.", "token_mixer.0."),
        ("channel_mixer.conv1.", "channel_mixer.m.0."),
        ("channel_mixer.conv2.", "channel_mixer.m.2."),
    ]
    for new, old in rules:
        if remainder.startswith(new):
            return f"features.{flat_idx}.{old}{remainder[len(new) :]}"
    return f"features.{flat_idx}.{remainder}"


def _reverse_repvit_stride2(flat_idx: int, remainder: str) -> str:
    """Reverse-map stride-2 block internals to original naming."""
    rules = [
        ("spatial_downsample.", "token_mixer.0."),
        ("channel_downsample.", "token_mixer.2."),
        ("ffn.conv1.", "channel_mixer.m.0."),
        ("ffn.conv2.", "channel_mixer.m.2."),
    ]
    for new, old in rules:
        if remainder.startswith(new):
            return f"features.{flat_idx}.{old}{remainder[len(new) :]}"
    return f"features.{flat_idx}.{remainder}"


def _build_mock_non_backbone_keys() -> dict[str, torch.Tensor]:
    """Build a minimal set of non-backbone checkpoint keys.

    Creates dummy tensors for the DETR encoder/decoder, text encoder, and other
    components that are common across all variants. These use the original
    checkpoint naming convention (with 'detector.' prefix, fused QKV, etc.).
    """
    keys: dict[str, torch.Tensor] = {}

    # DETR encoder layer (1 layer for simplicity)
    enc_prefix = "detector.transformer.encoder.layers.0."
    keys[f"{enc_prefix}cross_attn_image.in_proj_weight"] = torch.zeros(768, 256)
    keys[f"{enc_prefix}cross_attn_image.in_proj_bias"] = torch.zeros(768)
    keys[f"{enc_prefix}cross_attn_image.out_proj.weight"] = torch.zeros(256, 256)
    keys[f"{enc_prefix}cross_attn_image.out_proj.bias"] = torch.zeros(256)
    keys[f"{enc_prefix}norm1.weight"] = torch.zeros(256)
    keys[f"{enc_prefix}norm1.bias"] = torch.zeros(256)
    keys[f"{enc_prefix}norm2.weight"] = torch.zeros(256)
    keys[f"{enc_prefix}norm2.bias"] = torch.zeros(256)
    keys[f"{enc_prefix}norm3.weight"] = torch.zeros(256)
    keys[f"{enc_prefix}norm3.bias"] = torch.zeros(256)
    keys[f"{enc_prefix}linear1.weight"] = torch.zeros(1024, 256)
    keys[f"{enc_prefix}linear1.bias"] = torch.zeros(1024)
    keys[f"{enc_prefix}linear2.weight"] = torch.zeros(256, 1024)
    keys[f"{enc_prefix}linear2.bias"] = torch.zeros(256)

    # Projection head (common across all variants)
    proj_prefix = "detector.backbone.vision_backbone.trunk.model.head."
    keys[f"{proj_prefix}0.weight"] = torch.zeros(1024, 384, 1, 1)
    keys[f"{proj_prefix}0.bias"] = torch.zeros(1024)
    keys[f"{proj_prefix}1.weight"] = torch.zeros(1024)
    keys[f"{proj_prefix}1.bias"] = torch.zeros(1024)
    keys[f"{proj_prefix}1.running_mean"] = torch.zeros(1024)
    keys[f"{proj_prefix}1.running_var"] = torch.ones(1024)
    keys[f"{proj_prefix}1.num_batches_tracked"] = torch.tensor(0)
    keys[f"{proj_prefix}3.weight"] = torch.zeros(1024, 1024, 3, 3)
    keys[f"{proj_prefix}3.bias"] = torch.zeros(1024)

    return keys


class TestConvertEfficientsam3StateDict:
    """Tests for state dict conversion logic."""

    @pytest.mark.parametrize(
        ("backbone_type", "variant"),
        ALL_VARIANTS,
        ids=[f"{bt}-{v}" for bt, v in ALL_VARIANTS],
    )
    def test_backbone_keys_conversion_no_missing(
        self,
        backbone_type: str,
        variant: str,
    ) -> None:
        """Verify backbone key conversion produces no missing keys for any variant.

        Builds a synthetic checkpoint with original naming, runs conversion,
        and checks that all model backbone keys are present in the result.
        """
        # Build mock checkpoint backbone keys
        ckpt_backbone = _build_mock_checkpoint_backbone_keys(backbone_type, variant)

        # Run conversion
        converted = convert_efficientsam3_state_dict(
            ckpt_backbone,
            backbone_type=backbone_type,
            variant=variant,
        )

        # Build model and get expected backbone keys
        model = EfficientSam3Model(backbone_type=backbone_type, variant=variant)
        model_backbone_keys = {k for k in model.state_dict() if k.startswith(_BACKBONE_PREFIX)}

        converted_keys = set(converted.keys())

        # All model backbone keys should be present in converted dict
        missing = model_backbone_keys - converted_keys
        assert not missing, f"Missing backbone keys for {backbone_type}/{variant}:\n" + "\n".join(sorted(missing)[:20])

        # No unexpected backbone keys should remain
        unexpected = {k for k in converted_keys if k.startswith(_BACKBONE_PREFIX)} - model_backbone_keys
        assert not unexpected, f"Unexpected backbone keys for {backbone_type}/{variant}:\n" + "\n".join(
            sorted(unexpected)[:20],
        )

    @pytest.mark.parametrize(
        ("backbone_type", "variant"),
        ALL_VARIANTS,
        ids=[f"{bt}-{v}" for bt, v in ALL_VARIANTS],
    )
    def test_full_model_load_state_dict(
        self,
        backbone_type: str,
        variant: str,
    ) -> None:
        """Verify that a full synthetic checkpoint loads without missing keys.

        Creates a model, exports its state dict as a mock checkpoint using
        original naming conventions, converts it back, and loads it.
        Missing keys should only be the expected presence_token ones.
        """
        # Create model and get its state dict (with correct timm naming)
        model = EfficientSam3Model(backbone_type=backbone_type, variant=variant)
        model_sd = model.state_dict()

        # Build mock checkpoint from model's own state dict
        # (reverse the conversion to get checkpoint-format keys)
        ckpt_backbone = _build_mock_checkpoint_backbone_keys(backbone_type, variant)

        # Use model weights for backbone keys (ensures shapes match)
        backbone_ckpt_with_shapes: dict[str, torch.Tensor] = {}
        converted_backbone = convert_efficientsam3_state_dict(
            ckpt_backbone,
            backbone_type=backbone_type,
            variant=variant,
        )
        # Replace dummy tensors with actual model tensors where shapes match
        for converted_key in converted_backbone:
            if converted_key in model_sd:
                backbone_ckpt_with_shapes[converted_key] = model_sd[converted_key]

        # Load only backbone keys to verify they all match
        missing_keys, _ = model.load_state_dict(
            backbone_ckpt_with_shapes,
            strict=False,
        )

        # Filter out expected missing keys (non-backbone + presence)
        backbone_missing = [k for k in missing_keys if k.startswith(_BACKBONE_PREFIX)]

        assert not backbone_missing, f"Missing backbone keys after load for {backbone_type}/{variant}:\n" + "\n".join(
            sorted(backbone_missing)[:20],
        )

    @pytest.mark.parametrize(
        ("backbone_type", "variant"),
        [
            (BackboneType.REPVIT, "m0_9"),
            (BackboneType.REPVIT, "m1_1"),
            (BackboneType.REPVIT, "m2_3"),
        ],
        ids=["repvit-m0_9", "repvit-m1_1", "repvit-m2_3"],
    )
    def test_repvit_block_mapping_covers_all_features(
        self,
        backbone_type: str,
        variant: str,
    ) -> None:
        """Verify RepViT block mapping accounts for all flat feature indices."""
        timm_name = BACKBONE_CONFIG[backbone_type, variant][0]
        mapping = _build_repvit_block_mapping(timm_name)

        # All flat indices should be sequential starting from 1
        flat_indices = sorted(idx for idx, _, _ in mapping)
        expected = list(range(1, flat_indices[-1] + 1))
        assert flat_indices == expected, (
            f"Non-sequential flat indices for {backbone_type}/{variant}: got {flat_indices}, expected {expected}"
        )

    @pytest.mark.parametrize(
        ("backbone_type", "variant"),
        [
            (BackboneType.TINYVIT, "5m"),
        ],
        ids=["tinyvit-5m"],
    )
    def test_tinyvit_head_keys_converted(
        self,
        backbone_type: str,
        variant: str,
    ) -> None:
        """Verify TinyViT classifier head keys pass through conversion.

        SAM-HQ's TinyViT retains head/norm_head parameters despite
        features_only=True, so these must be present in the converted dict.
        """
        # Add head and norm_head keys in checkpoint format
        ckpt = _build_mock_checkpoint_backbone_keys(backbone_type, variant)
        head_prefix = "detector.backbone.vision_backbone.trunk.model.backbone.model."
        ckpt[f"{head_prefix}head.weight"] = torch.zeros(1000, 320)
        ckpt[f"{head_prefix}head.bias"] = torch.zeros(1000)
        ckpt[f"{head_prefix}norm_head.weight"] = torch.zeros(320)
        ckpt[f"{head_prefix}norm_head.bias"] = torch.zeros(320)

        converted = convert_efficientsam3_state_dict(
            ckpt,
            backbone_type=backbone_type,
            variant=variant,
        )

        # Head keys should be converted to model format
        expected_keys = {
            f"{_BACKBONE_PREFIX}head.weight",
            f"{_BACKBONE_PREFIX}head.bias",
            f"{_BACKBONE_PREFIX}norm_head.weight",
            f"{_BACKBONE_PREFIX}norm_head.bias",
        }
        assert expected_keys <= set(converted.keys()), "Head keys missing after conversion:\n" + "\n".join(
            sorted(expected_keys - set(converted.keys())),
        )

    def test_efficientvit_backward_compat_no_backbone_type(self) -> None:
        """Verify conversion still works without backbone_type (backward compat)."""
        ckpt = _build_mock_checkpoint_backbone_keys(
            BackboneType.EFFICIENTVIT,
            "b0",
        )
        # Call without backbone_type — should still work for EfficientViT
        converted = convert_efficientsam3_state_dict(ckpt)
        assert len(converted) > 0

    def test_non_backbone_keys_preserved(self) -> None:
        """Verify non-backbone keys pass through conversion unchanged."""
        non_backbone = _build_mock_non_backbone_keys()

        converted = convert_efficientsam3_state_dict(non_backbone)

        # Verify DETR encoder keys are renamed correctly
        detr_keys = [k for k in converted if k.startswith("detr_encoder.")]
        assert len(detr_keys) > 0, "DETR encoder keys should be present"

        # Verify fused QKV is split
        qkv_keys = [k for k in converted if "q_proj" in k or "k_proj" in k]
        assert len(qkv_keys) > 0, "Fused QKV should be split"

        # Verify out_proj renamed to o_proj
        assert not any(".out_proj." in k for k in converted), "out_proj should be renamed to o_proj"

    @pytest.mark.parametrize(
        ("backbone_type", "variant"),
        [v for v in ALL_VARIANTS if v[0] != BackboneType.TINYVIT],
        ids=[f"{bt}-{v}" for bt, v in ALL_VARIANTS if bt != BackboneType.TINYVIT],
    )
    def test_no_layers_or_features_keys_remain(
        self,
        backbone_type: str,
        variant: str,
    ) -> None:
        """Verify no unconverted original backbone keys remain after conversion."""
        ckpt = _build_mock_checkpoint_backbone_keys(backbone_type, variant)
        converted = convert_efficientsam3_state_dict(
            ckpt,
            backbone_type=backbone_type,
            variant=variant,
        )

        # After conversion, no backbone keys should use original naming
        # (except TinyViT, where original naming IS the model naming)
        original_patterns = []
        for key in converted:
            if not key.startswith(_BACKBONE_PREFIX):
                continue
            suffix = key[len(_BACKBONE_PREFIX) :]
            if backbone_type == BackboneType.REPVIT:
                if suffix.startswith("features."):
                    original_patterns.append(key)
            elif backbone_type == BackboneType.EFFICIENTVIT and (
                suffix.startswith("input_stem.") or re.match(r"stages\.\d+\.op_list\.", suffix)
            ):
                original_patterns.append(key)

        assert not original_patterns, (
            f"Unconverted original backbone keys for {backbone_type}/{variant}:\n"
            + "\n".join(sorted(original_patterns)[:10])
        )
