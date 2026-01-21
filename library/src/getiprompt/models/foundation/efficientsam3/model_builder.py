# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientSAM3 model builder.

This module provides factory functions to build EfficientSAM3 models with
student backbones (RepViT, TinyViT) instead of the full ViT.

EfficientSAM3 achieves faster inference through knowledge distillation,
using lightweight backbones while maintaining segmentation quality.

Supported backbones:
    - RepViT: repvit-m0.9, repvit-m1.1, repvit-m2.3
    - TinyViT: tinyvit-5m, tinyvit-11m, tinyvit-21m

Supported text encoders:
    - sam3-full: Full SAM3 text encoder (highest quality)
    - MobileCLIP-S0/S1/B: Efficient student text encoders


Example:
    >>> from getiprompt.models.foundation.efficientsam3 import (
    ...     build_efficientsam3_image_model,
    ...     EfficientSAM3BackboneType,
    ... )
    >>> model = build_efficientsam3_image_model(
    ...     backbone_type=EfficientSAM3BackboneType.TINYVIT_21M,
    ...     device="cuda",
    ... )
"""

from enum import StrEnum
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn

# Import SAM3 shared components
from getiprompt.models.foundation.sam3.model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from getiprompt.models.foundation.sam3.model.encoder import (
    TransformerEncoderFusion,
    TransformerEncoderLayer,
)
from getiprompt.models.foundation.sam3.model.geometry_encoders import (
    SequenceGeometryEncoder,
)
from getiprompt.models.foundation.sam3.model.maskformer_segmentation import (
    PixelDecoder,
    UniversalSegmentationHead,
)
from getiprompt.models.foundation.sam3.model.model_misc import (
    MLP,
    DotProductScoring,
    TransformerWrapper,
)
from getiprompt.models.foundation.sam3.model.model_misc import (
    MultiheadAttentionWrapper as MultiheadAttention,
)
from getiprompt.models.foundation.sam3.model.necks import Sam3DualViTDetNeck
from getiprompt.models.foundation.sam3.model.position_encoding import (
    PositionEmbeddingSine,
)
from getiprompt.models.foundation.sam3.model.text_encoder_ve import VETextEncoder
from getiprompt.models.foundation.sam3.model.tokenizer_ve import SimpleTokenizer
from getiprompt.models.foundation.sam3.model.vl_combiner import SAM3VLBackbone
from getiprompt.models.foundation.sam3.sam.mask_decoder import MaskDecoder
from getiprompt.models.foundation.sam3.sam.prompt_encoder import PromptEncoder
from getiprompt.models.foundation.sam3.sam.transformer import TwoWayTransformer

from .efficientsam3_image import EfficientSAM3Image

__all__ = [
    "EfficientSAM3BackboneType",
    "EfficientSAM3TextEncoderType",
    "ImageStudentEncoder",
    "build_efficientsam3_image_model",
]


class EfficientSAM3BackboneType(StrEnum):
    """Available backbone types for EfficientSAM3."""

    # RepViT variants
    REPVIT_M0_9 = "repvit-m0.9"
    REPVIT_M1_1 = "repvit-m1.1"
    REPVIT_M2_3 = "repvit-m2.3"

    # TinyViT variants
    TINYVIT_5M = "tinyvit-5m"
    TINYVIT_11M = "tinyvit-11m"
    TINYVIT_21M = "tinyvit-21m"

    # EfficientViT variants
    EFFICIENTVIT_B0 = "efficientvit-b0"
    EFFICIENTVIT_B1 = "efficientvit-b1"
    EFFICIENTVIT_B2 = "efficientvit-b2"


class EfficientSAM3TextEncoderType(StrEnum):
    """Available text encoder types for EfficientSAM3."""

    # MobileCLIP variants (compact student text encoders)
    MOBILECLIP_S0 = "MobileCLIP-S0"  # 4 layers, smallest
    MOBILECLIP_S1 = "MobileCLIP-S1"  # 12 layers, default
    MOBILECLIP_B = "MobileCLIP-B"  # 12 layers, causal masking
    MOBILECLIP2_L = "MobileCLIP2-L"  # Larger MobileCLIP2 variant


# ==============================================================================
# Model Checkpoint Registry
# ==============================================================================

EFFICIENTSAM3_HF_REPO = "Simon7108528/EfficientSAM3"
EFFICIENTSAM3_HF_SUBFOLDER = "stage1_all_converted"

# Registry mapping (backbone_type, text_encoder_type) to checkpoint filename
# Note: text_encoder_type=None means SAM3 full text encoder (from SAM3 codebase)
MODEL_CONFIGS = {
    # SAM3 Full Text Encoder + EfficientSAM3 Image Encoder Models (image encoder only checkpoints)
    (EfficientSAM3BackboneType.REPVIT_M0_9, None): "efficient_sam3_repvit_s.pt",
    (EfficientSAM3BackboneType.REPVIT_M1_1, None): "efficient_sam3_repvit_m.pt",
    (EfficientSAM3BackboneType.REPVIT_M2_3, None): "efficient_sam3_repvit_l.pt",
    (EfficientSAM3BackboneType.TINYVIT_5M, None): "efficient_sam3_tinyvit_s.pt",
    (EfficientSAM3BackboneType.TINYVIT_11M, None): "efficient_sam3_tinyvit_m.pt",
    (EfficientSAM3BackboneType.TINYVIT_21M, None): "efficient_sam3_tinyvit_l.pt",
    (EfficientSAM3BackboneType.EFFICIENTVIT_B0, None): "efficient_sam3_efficientvit_s.pt",
    (EfficientSAM3BackboneType.EFFICIENTVIT_B1, None): "efficient_sam3_efficientvit_m.pt",
    (EfficientSAM3BackboneType.EFFICIENTVIT_B2, None): "efficient_sam3_efficientvit_l.pt",
    # MobileCLIP-S1 Text Encoder + EfficientSAM3 Image Encoder Models (unified checkpoints)
    (
        EfficientSAM3BackboneType.REPVIT_M0_9,
        EfficientSAM3TextEncoderType.MOBILECLIP_S1,
    ): "efficient_sam3_repvit-m0_9_mobileclip_s1.pth",
    (
        EfficientSAM3BackboneType.REPVIT_M1_1,
        EfficientSAM3TextEncoderType.MOBILECLIP_S1,
    ): "efficient_sam3_repvit-m1_1_mobileclip_s1.pth",
    (
        EfficientSAM3BackboneType.REPVIT_M2_3,
        EfficientSAM3TextEncoderType.MOBILECLIP_S1,
    ): "efficient_sam3_repvit-m2_3_mobileclip_s1.pth",
    (
        EfficientSAM3BackboneType.TINYVIT_5M,
        EfficientSAM3TextEncoderType.MOBILECLIP_S1,
    ): "efficient_sam3_tinyvit_5m_mobileclip_s1.pth",
    (
        EfficientSAM3BackboneType.TINYVIT_11M,
        EfficientSAM3TextEncoderType.MOBILECLIP_S1,
    ): "efficient_sam3_tinyvit_11m_mobileclip_s1.pth",
    (
        EfficientSAM3BackboneType.TINYVIT_21M,
        EfficientSAM3TextEncoderType.MOBILECLIP_S1,
    ): "efficient_sam3_tinyvit_21m_mobileclip_s1.pth",
    (
        EfficientSAM3BackboneType.EFFICIENTVIT_B0,
        EfficientSAM3TextEncoderType.MOBILECLIP_S1,
    ): "efficient_sam3_efficientvit-b0_mobileclip_s1.pth",
    (
        EfficientSAM3BackboneType.EFFICIENTVIT_B1,
        EfficientSAM3TextEncoderType.MOBILECLIP_S1,
    ): "efficient_sam3_efficientvit-b1_mobileclip_s1.pth",
    (
        EfficientSAM3BackboneType.EFFICIENTVIT_B2,
        EfficientSAM3TextEncoderType.MOBILECLIP_S1,
    ): "efficient_sam3_efficientvit-b2_mobileclip_s1.pth",
}


# ==============================================================================
# Shared component builders (reused from SAM3)
# ==============================================================================


def _create_position_encoding(
    precompute_resolution: int | None = None,
    device: str = "cuda",
) -> PositionEmbeddingSine:
    """Create position encoding for visual backbone."""
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
        device=device,
    )


def _create_vit_neck(
    position_encoding: PositionEmbeddingSine,
    backbone: nn.Module,
    enable_inst_interactivity: bool = False,
) -> Sam3DualViTDetNeck:
    """Create ViT neck for feature pyramid."""
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=backbone,
        add_sam2_neck=enable_inst_interactivity,
    )


def _create_vl_backbone(
    vit_neck: Sam3DualViTDetNeck,
    text_encoder: nn.Module,
) -> SAM3VLBackbone:
    """Create visual-language backbone."""
    return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)


def _create_transformer_encoder() -> TransformerEncoderFusion:
    """Create transformer encoder with its layer."""
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
    )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    return encoder


def _create_transformer_decoder() -> TransformerDecoder:
    """Create transformer decoder with its layer."""
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )
    return decoder


def _create_dot_product_scoring() -> DotProductScoring:
    """Create dot product scoring module."""
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


def _create_segmentation_head(
    compile_mode: str | None = None,
) -> UniversalSegmentationHead:
    """Create segmentation head with pixel decoder."""
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
        compile_mode=compile_mode,
    )

    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dropout=0,
        embed_dim=256,
    )

    segmentation_head = UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )
    return segmentation_head


def _create_geometry_encoder(device: str = "cuda") -> SequenceGeometryEncoder:
    """Create geometry encoder with all its components."""
    geo_pos_enc = _create_position_encoding(device=device)

    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )

    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )
    return input_geometry_encoder


def _create_sam3_transformer() -> TransformerWrapper:
    """Create SAM3 transformer encoder and decoder."""
    encoder = _create_transformer_encoder()
    decoder = _create_transformer_decoder()
    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


def _create_sam3_model(
    backbone: SAM3VLBackbone,
    transformer: TransformerWrapper,
    input_geometry_encoder: SequenceGeometryEncoder,
    segmentation_head: UniversalSegmentationHead | None,
    dot_prod_scoring: DotProductScoring,
) -> EfficientSAM3Image:
    """Create the EfficientSAM3 image model.

    Returns EfficientSAM3Image which extends Sam3Image with predict_inst methods.
    """
    return EfficientSAM3Image(
        backbone=backbone,
        transformer=transformer,
        input_geometry_encoder=input_geometry_encoder,
        segmentation_head=segmentation_head,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=dot_prod_scoring,
        use_instance_query=False,
        multimask_output=True,
        matcher=None,
    )


def _add_sam_heads(
    model: EfficientSAM3Image,
    image_size: int = 1008,
    backbone_stride: int = 14,
    device: str = "cuda",
) -> None:
    """Add SAM-style prompt encoder and mask decoder to the model.

    Args:
        model: The EfficientSAM3 model to add heads to
        image_size: Input image size (default 1008)
        backbone_stride: Backbone output stride (default 14)
        device: Device to place the heads on
    """
    hidden_dim = model.hidden_dim
    sam_image_embedding_size = image_size // backbone_stride
    num_feature_levels = 3  # SAM uses 3 feature levels

    model.image_size = image_size
    model.backbone_stride = backbone_stride
    model.sam_prompt_embed_dim = hidden_dim
    model.sam_image_embedding_size = sam_image_embedding_size

    # Add attributes needed by SAM1 predictor
    model.directly_add_no_mem_embed = False
    model.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, hidden_dim).to(device))
    model._bb_feat_sizes = [(288, 288), (144, 144), (72, 72)]

    # Build PromptEncoder and MaskDecoder from SAM
    model.sam_prompt_encoder = PromptEncoder(
        embed_dim=hidden_dim,
        image_embedding_size=(sam_image_embedding_size, sam_image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    ).to(device)

    model.sam_mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=hidden_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=hidden_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        use_high_res_features=True,  # Enable conv_s0 and conv_s1 for high-res features
        iou_prediction_use_sigmoid=True,  # Apply sigmoid to IOU predictions (0-1 range)
        pred_obj_scores=True,  # Predict object scores
        pred_obj_scores_mlp=True,  # Use MLP for object score prediction
        use_multimask_token_for_obj_ptr=True,  # Use multimask token for object pointer
    ).to(device)

    # Add forward_image wrapper
    def forward_image(img_batch):
        return model.backbone.forward_image(img_batch)

    model.forward_image = forward_image

    # Add _prepare_backbone_features wrapper
    # This matches the sam3_tracker_base.py implementation
    def _prepare_backbone_features(backbone_out):
        """Extract and prepare features from backbone output.

        Returns vision_feats in flattened format: HWxNxC for each feature level.
        """
        backbone_out = backbone_out.copy() if isinstance(backbone_out, dict) else backbone_out

        feature_maps = backbone_out["backbone_fpn"][-num_feature_levels:]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-num_feature_levels:]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]

        # Flatten NxCxHxW to HWxNxC (same as sam3_tracker_base)
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    model._prepare_backbone_features = _prepare_backbone_features


def _create_text_encoder(bpe_path: str | Path) -> VETextEncoder:
    """Create SAM3 text encoder."""
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return VETextEncoder(
        tokenizer=tokenizer,
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
    )


def _create_student_text_encoder(
    bpe_path: str | Path,
    text_encoder_type: EfficientSAM3TextEncoderType,
) -> nn.Module:
    """Create student text encoder (MobileCLIP variants)."""
    from getiprompt.models.foundation.efficientsam3.backbones.mobile_clip import (
        TextStudentEncoder,
    )

    # Default config values
    cfg = {
        "context_length": 77,
        "vocab_size": 49408,
        "dim": 512,
        "ffn_multiplier_per_layer": 4.0,
        "n_heads_per_layer": 8,
        "n_transformer_layers": 12,
        "norm_layer": "layer_norm_fp32",
        "causal_masking": False,
        "model_name": "base",
        "embed_dropout": 0.0,
        "no_scale_embedding": False,
        "no_pos_embedding": False,
    }

    if text_encoder_type == EfficientSAM3TextEncoderType.MOBILECLIP_S0:
        cfg.update(
            {
                "dim": 512,
                "n_transformer_layers": 4,
                "n_heads_per_layer": 8,
                "model_name": "mct",
            },
        )
    elif text_encoder_type == EfficientSAM3TextEncoderType.MOBILECLIP_S1:
        cfg.update(
            {
                "dim": 512,
                "n_transformer_layers": 12,
                "n_heads_per_layer": 8,
                "model_name": "base",
            },
        )
    elif text_encoder_type == EfficientSAM3TextEncoderType.MOBILECLIP_B:
        cfg.update(
            {
                "dim": 512,
                "n_transformer_layers": 12,
                "n_heads_per_layer": 8,
                "model_name": "base",
                "causal_masking": True,
            },
        )
    elif text_encoder_type == EfficientSAM3TextEncoderType.MOBILECLIP2_L:
        cfg.update(
            {
                "dim": 768,
                "n_transformer_layers": 12,
                "n_heads_per_layer": 12,
                "model_name": "large",
            },
        )

    return TextStudentEncoder(
        cfg=cfg,
        context_length=32,  # Match teacher input length
        output_dim=256,  # SAM3 d_model
        bpe_path=bpe_path,
    )


# ==============================================================================
# Student Vision Backbone
# ==============================================================================


class ImageStudentEncoder(nn.Module):
    """Projection head for student backbones to match SAM3's expected dimensions.

    This module wraps a lightweight backbone (EfficientViT, RepViT, TinyViT) and
    projects its output to match SAM3's expected feature dimensions (1024 channels
    at 72x72 resolution for 1008x1008 input).
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_channels: int,
        embed_dim: int = 1024,
        embed_size: int = 72,
        img_size: int = 1008,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embed_size = embed_size
        self.img_size = img_size
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = self.head(feats)
        if feats.shape[-1] != self.embed_size or feats.shape[-2] != self.embed_size:
            feats = F.interpolate(
                feats,
                size=(self.embed_size, self.embed_size),
                mode="bilinear",
                align_corners=False,
            )
        return feats


def _create_student_vision_backbone(
    backbone_type: EfficientSAM3BackboneType,
    enable_inst_interactivity: bool = False,
    device: str = "cuda",
) -> Sam3DualViTDetNeck:
    """Create EfficientSAM3 visual backbone with a student backbone and neck.

    Args:
        backbone_type: Type of student backbone to use.
        enable_inst_interactivity: Whether to enable SAM2-style neck for tracking.
        device: Target device.

    Returns:
        Sam3DualViTDetNeck with student encoder.
    """
    position_encoding = _create_position_encoding(precompute_resolution=1008, device=device)

    # Create backbone based on type
    if backbone_type in (
        EfficientSAM3BackboneType.REPVIT_M0_9,
        EfficientSAM3BackboneType.REPVIT_M1_1,
        EfficientSAM3BackboneType.REPVIT_M2_3,
    ):
        from getiprompt.models.foundation.efficientsam3.backbones.repvit import (
            repvit_m0_9,
            repvit_m1_1,
            repvit_m2_3,
        )

        backbone_map = {
            EfficientSAM3BackboneType.REPVIT_M0_9: repvit_m0_9,
            EfficientSAM3BackboneType.REPVIT_M1_1: repvit_m1_1,
            EfficientSAM3BackboneType.REPVIT_M2_3: repvit_m2_3,
        }
        backbone = backbone_map[backbone_type](distillation=False, num_classes=0)

        class RepViTTrunkWrapper(nn.Module):
            """Wrapper to extract features from RepViT."""

            def __init__(self, model: nn.Module) -> None:
                super().__init__()
                self.model = model
                # Infer channels from a forward pass
                dummy = torch.zeros(1, 3, 224, 224)
                with torch.no_grad():
                    for f in model.features:
                        dummy = f(dummy)
                self.channel_list = [dummy.shape[1]]

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for f in self.model.features:
                    x = f(x)
                return x

        wrapped_backbone = RepViTTrunkWrapper(backbone)
        in_channels = wrapped_backbone.channel_list[0]

    elif backbone_type in (
        EfficientSAM3BackboneType.TINYVIT_5M,
        EfficientSAM3BackboneType.TINYVIT_11M,
        EfficientSAM3BackboneType.TINYVIT_21M,
    ):
        from getiprompt.models.foundation.efficientsam3.backbones.tiny_vit import (
            tiny_vit_5m_224,
            tiny_vit_11m_224,
            tiny_vit_21m_224,
        )

        backbone_map = {
            EfficientSAM3BackboneType.TINYVIT_5M: tiny_vit_5m_224,
            EfficientSAM3BackboneType.TINYVIT_11M: tiny_vit_11m_224,
            EfficientSAM3BackboneType.TINYVIT_21M: tiny_vit_21m_224,
        }
        backbone = backbone_map[backbone_type](img_size=1008, num_classes=0)

        class TinyViTTrunkWrapper(nn.Module):
            """Wrapper to extract features from TinyViT."""

            def __init__(self, model: nn.Module) -> None:
                super().__init__()
                self.model = model
                self.channel_list = [model.layers[-1].dim]

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.model.patch_embed(x)
                for layer in self.model.layers:
                    x = layer(x)
                # Reshape from (B, L, C) to (B, C, H, W)
                B, L, C = x.shape
                H, W = self.model.layers[-1].input_resolution
                x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                return x

        wrapped_backbone = TinyViTTrunkWrapper(backbone)
        in_channels = wrapped_backbone.channel_list[0]

    elif backbone_type in (
        EfficientSAM3BackboneType.EFFICIENTVIT_B0,
        EfficientSAM3BackboneType.EFFICIENTVIT_B1,
        EfficientSAM3BackboneType.EFFICIENTVIT_B2,
    ):
        from getiprompt.models.foundation.efficientsam3.backbones.efficientvit import (
            efficientvit_b0,
            efficientvit_b1,
            efficientvit_b2,
        )

        backbone_map = {
            EfficientSAM3BackboneType.EFFICIENTVIT_B0: efficientvit_b0,
            EfficientSAM3BackboneType.EFFICIENTVIT_B1: efficientvit_b1,
            EfficientSAM3BackboneType.EFFICIENTVIT_B2: efficientvit_b2,
        }
        backbone = backbone_map[backbone_type](img_size=1008)

        class EfficientViTTrunkWrapper(nn.Module):
            """Wrapper to extract features from EfficientViT."""

            def __init__(self, model: nn.Module) -> None:
                super().__init__()
                self.model = model
                self.channel_list = [model.num_features]

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.model.patch_embed(x)
                for stage in self.model.stages:
                    x = stage(x)
                return x

        wrapped_backbone = EfficientViTTrunkWrapper(backbone)
        in_channels = wrapped_backbone.channel_list[0]

    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")

    # Wrap with ImageStudentEncoder for projection
    student_encoder = ImageStudentEncoder(
        backbone=wrapped_backbone,
        in_channels=in_channels,
        embed_dim=1024,  # SAM3 expects 1024 channels
        embed_size=72,
        img_size=1008,
    )

    # Add channel_list for Sam3DualViTDetNeck compatibility
    student_encoder.channel_list = [1024]

    # Wrap to match Sam3DualViTDetNeck's trunk interface
    # The neck expects trunk.forward to accept a tensor and return a tensor
    # (despite the parameter being named tensor_list)
    final_trunk = student_encoder

    vit_neck = _create_vit_neck(
        position_encoding,
        final_trunk,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    return vit_neck


# ==============================================================================
# Checkpoint loading
# ==============================================================================


def _load_checkpoint(
    model: EfficientSAM3Image,
    checkpoint_path: str | Path,
    enable_inst_interactivity: bool = False,
) -> None:
    """Load model checkpoint from file.

    This is a wrapper around the new checkpoint loader system for backward compatibility.

    Args:
        model: Model to load checkpoint into.
        checkpoint_path: Path to the checkpoint file.
        enable_inst_interactivity: Whether to load tracker/SAM heads weights.
    """
    from getiprompt.models.foundation.efficientsam3.efficientsam3_checkpoint_loaders import (
        EfficientSAM3CheckpointPaths,
        EfficientSAM3UnifiedCheckpointLoader,
    )

    loader = EfficientSAM3UnifiedCheckpointLoader()
    paths = EfficientSAM3CheckpointPaths(unified=Path(checkpoint_path))
    loader.load(model, paths, enable_inst_interactivity)


def _setup_device_and_mode(
    model: EfficientSAM3Image,
    device: str,
    eval_mode: bool,
) -> EfficientSAM3Image:
    """Setup model device and evaluation mode."""
    if device == "cuda":
        model = model.cuda()
    elif device == "xpu":
        model = model.to("xpu")
    if eval_mode:
        model.eval()
    return model


# ==============================================================================
# Main builder function
# ==============================================================================


def get_checkpoint_filename(
    backbone_type: EfficientSAM3BackboneType,
    text_encoder_type: EfficientSAM3TextEncoderType | None,
) -> str:
    """Get checkpoint filename for given backbone and text encoder combination.

    Args:
        backbone_type: Type of image encoder backbone
        text_encoder_type: Type of text encoder (None for SAM3 full text encoder)

    Returns:
        Checkpoint filename

    Raises:
        ValueError: If the combination is not available
    """
    key = (backbone_type, text_encoder_type)
    if key not in MODEL_CONFIGS:
        available = "\n".join(f"  - {bb.value} + {te.value if te else 'SAM3-full'}" for bb, te in MODEL_CONFIGS)
        raise ValueError(
            f"Model combination not available: {backbone_type.value} + "
            f"{text_encoder_type.value if text_encoder_type else 'SAM3-full'}\n"
            f"Available combinations:\n{available}",
        )
    return MODEL_CONFIGS[key]


def download_efficientsam3_from_hf(
    backbone_type: EfficientSAM3BackboneType,
    text_encoder_type: EfficientSAM3TextEncoderType | None = None,
) -> str:
    """Download EfficientSAM3 checkpoint from HuggingFace Hub.

    Args:
        backbone_type: Type of student backbone
        text_encoder_type: Type of text encoder (None for SAM3 full encoder)

    Returns:
        Path to downloaded checkpoint file

    Raises:
        ValueError: If the model combination is not available
    """
    checkpoint_filename = get_checkpoint_filename(backbone_type, text_encoder_type)

    print(
        f"Downloading EfficientSAM3 checkpoint: {checkpoint_filename} from HuggingFace Hub...\n"
        f"Repository: {EFFICIENTSAM3_HF_REPO}/{EFFICIENTSAM3_HF_SUBFOLDER}",
    )

    checkpoint_path = hf_hub_download(
        repo_id=EFFICIENTSAM3_HF_REPO,
        filename=checkpoint_filename,
        subfolder=EFFICIENTSAM3_HF_SUBFOLDER,
    )

    print(f"Checkpoint downloaded to: {checkpoint_path}")
    return checkpoint_path


def build_efficientsam3_image_model(
    bpe_path: str | Path | None = None,
    device: str = "cuda",
    checkpoint_path: str | Path | None = None,
    load_from_HF: bool = True,
    enable_segmentation: bool = True,
    enable_inst_interactivity: bool = False,
    compile: bool = False,
    backbone_type: EfficientSAM3BackboneType | str = EfficientSAM3BackboneType.TINYVIT_21M,
    text_encoder_type: EfficientSAM3TextEncoderType | str | None = EfficientSAM3TextEncoderType.MOBILECLIP_S1,
) -> EfficientSAM3Image:
    """Build EfficientSAM3 image model with a student backbone.

    This creates a SAM3-compatible model but with a lightweight student backbone
    (EfficientViT, RepViT, or TinyViT) instead of the full ViT.

    Args:
        bpe_path: Path to BPE tokenizer vocabulary.
        device: Device to load the model on ('cuda', 'xpu', or 'cpu').
        checkpoint_path: Optional path to EfficientSAM3 model checkpoint.
        enable_segmentation: Whether to enable segmentation head.
        enable_inst_interactivity: Whether to enable instance interactivity.
        compile: Whether to compile the model with torch.compile.
        backbone_type: Type of student backbone to use.
        text_encoder_type: Type of text encoder. If None, uses full SAM3 encoder.

    Returns:
        An EfficientSAM3 image model (EfficientSAM3Image instance).

    Examples:
        >>> # Create EfficientSAM3 with TinyViT-21M backbone
        >>> model = build_efficientsam3_image_model(
        ...     backbone_type=EfficientSAM3BackboneType.TINYVIT_21M,
        ...     device="cuda",
        ... )

        >>> # Create with RepViT backbone and MobileCLIP text encoder
        >>> model = build_efficientsam3_image_model(
        ...     backbone_type=EfficientSAM3BackboneType.REPVIT_M2_3,
        ...     text_encoder_type=EfficientSAM3TextEncoderType.MOBILECLIP_S1,
        ... )
    """
    # Convert string to enum if needed
    if isinstance(backbone_type, str):
        backbone_type = EfficientSAM3BackboneType(backbone_type)

    if text_encoder_type is not None and isinstance(text_encoder_type, str):
        text_encoder_type = EfficientSAM3TextEncoderType(text_encoder_type)

    # Set default BPE path
    if bpe_path is None:
        bpe_path = Path(__file__).parent.parent / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    if not Path(bpe_path).exists():
        raise FileNotFoundError(f"BPE path {bpe_path} does not exist")

    compile_mode = "default" if compile else None

    # Create vision encoder with student backbone
    vision_encoder = _create_student_vision_backbone(
        backbone_type=backbone_type,
        enable_inst_interactivity=enable_inst_interactivity,
        device=device,
    )

    # Create text encoder
    if text_encoder_type is None:
        # Use full SAM3 text encoder (from SAM3 codebase)
        text_encoder = _create_text_encoder(bpe_path)
    else:
        # Use MobileCLIP student text encoder
        text_encoder = _create_student_text_encoder(bpe_path, text_encoder_type)

    # Create visual-language backbone
    backbone = _create_vl_backbone(vision_encoder, text_encoder)

    # Create transformer components
    transformer = _create_sam3_transformer()

    # Create dot product scoring
    dot_prod_scoring = _create_dot_product_scoring()

    # Create segmentation head if enabled
    segmentation_head = _create_segmentation_head(compile_mode=compile_mode) if enable_segmentation else None

    # Create geometry encoder
    input_geometry_encoder = _create_geometry_encoder(device=device)

    # Create the model
    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
    )

    # Initialize SAM heads if instance interactivity is enabled
    # This must be done BEFORE loading checkpoint so tracker weights can be loaded
    if enable_inst_interactivity:
        # Add SAM heads (prompt encoder and mask decoder)
        _add_sam_heads(model, image_size=1008, backbone_stride=14, device=device)

    # Auto-download checkpoint from HuggingFace if needed
    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_efficientsam3_from_hf(
            backbone_type=backbone_type,
            text_encoder_type=text_encoder_type,
        )

    # Load checkpoint BEFORE eval mode
    # This is critical because TinyViT's Attention layer computes the 'ab' buffer
    # during train(mode=False), which requires attention_biases to be loaded first.
    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path, enable_inst_interactivity=enable_inst_interactivity)

    # Setup device and mode AFTER loading checkpoint
    model = _setup_device_and_mode(model, device, eval_mode=True)

    # Create the interactive predictor if instance interactivity is enabled
    # This must be done AFTER checkpoint loading so the model has the correct weights
    if enable_inst_interactivity:
        from getiprompt.models.foundation.sam3.model.sam1_task_predictor import (
            SAM3InteractiveImagePredictor,
        )

        # Create predictor
        predictor = SAM3InteractiveImagePredictor(
            sam_model=model,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )
        # Move predictor's transforms to device
        predictor._transforms = predictor._transforms.to(device)
        model.inst_interactive_predictor = predictor

    return model
