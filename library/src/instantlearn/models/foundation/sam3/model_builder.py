# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch import nn

from instantlearn.models.foundation.sam3.model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from instantlearn.models.foundation.sam3.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from instantlearn.models.foundation.sam3.model.geometry_encoders import SequenceGeometryEncoder
from instantlearn.models.foundation.sam3.model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from instantlearn.models.foundation.sam3.model.model_misc import (
    MLP,
    DotProductScoring,
    TransformerWrapper,
)
from instantlearn.models.foundation.sam3.model.model_misc import (
    MultiheadAttentionWrapper as MultiheadAttention,
)
from instantlearn.models.foundation.sam3.model.necks import Sam3DualViTDetNeck
from instantlearn.models.foundation.sam3.model.position_encoding import PositionEmbeddingSine
from instantlearn.models.foundation.sam3.model.text_encoder_ve import VETextEncoder
from instantlearn.models.foundation.sam3.model.tokenizer_ve import SimpleTokenizer
from instantlearn.models.foundation.sam3.model.vitdet import ViT
from instantlearn.models.foundation.sam3.model.vl_combiner import SAM3VLBackbone
from instantlearn.models.foundation.sam3.sam3_image import Sam3Image


# Setup TensorFloat-32 for Ampere GPUs if available
def _setup_tf32() -> None:
    """Enable TensorFloat-32 for Ampere GPUs if available."""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


_setup_tf32()


def _create_position_encoding(precompute_resolution=None, device="cuda") -> nn.Module:
    """Create position encoding for visual backbone."""
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
        device=device,
    )


def _create_vit_backbone(compile_mode: str | None = None) -> ViT:
    """Create ViT backbone for visual feature extraction."""
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
        use_act_checkpoint=False,
    )


def _create_vit_neck(
    position_encoding: nn.Module,
    vit_backbone: ViT,
    enable_inst_interactivity: bool = False,
) -> Sam3DualViTDetNeck:
    """Create ViT neck for feature pyramid."""
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )


def _create_vl_backbone(vit_neck: Sam3DualViTDetNeck, text_encoder: VETextEncoder) -> SAM3VLBackbone:
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

    return TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )


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

    return TransformerDecoder(
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


def _create_segmentation_head(compile_mode: str | None = None) -> nn.Module:
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

    return UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )


def _create_geometry_encoder(device: str = "cuda") -> nn.Module:
    """Create geometry encoder with all its components."""
    # Create position encoding for geometry encoder
    geo_pos_enc = _create_position_encoding(device=device)
    # Create geometry encoder layer
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

    # Create geometry encoder
    return SequenceGeometryEncoder(
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


def _create_sam3_model(
    backbone: SAM3VLBackbone,
    transformer: TransformerWrapper,
    input_geometry_encoder: nn.Module,
    segmentation_head: nn.Module | None,
    dot_prod_scoring: nn.Module | None,
) -> Sam3Image:
    """Create the SAM3 image model."""
    common_params = {
        "backbone": backbone,
        "transformer": transformer,
        "input_geometry_encoder": input_geometry_encoder,
        "segmentation_head": segmentation_head,
        "num_feature_levels": 1,
        "o2m_mask_predict": True,
        "dot_prod_scoring": dot_prod_scoring,
        "use_instance_query": False,
        "multimask_output": True,
        "matcher": None,
    }

    return Sam3Image(**common_params)


def _create_text_encoder(bpe_path: str) -> VETextEncoder:
    """Create SAM3 text encoder."""
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return VETextEncoder(
        tokenizer=tokenizer,
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
    )


def _create_vision_backbone(
    compile_mode: str | None = None,
    enable_inst_interactivity: bool = True,
    device: str = "cuda",
) -> Sam3DualViTDetNeck:
    """Create SAM3 visual backbone with ViT and neck."""
    # Position encoding
    position_encoding = _create_position_encoding(precompute_resolution=1008, device=device)
    # ViT backbone
    vit_backbone: ViT = _create_vit_backbone(compile_mode=compile_mode)
    vit_neck: Sam3DualViTDetNeck = _create_vit_neck(
        position_encoding,
        vit_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    # Visual neck
    return vit_neck


def _create_sam3_transformer() -> TransformerWrapper:
    """Create SAM3 transformer encoder and decoder."""
    encoder: TransformerEncoderFusion = _create_transformer_encoder()
    decoder: TransformerDecoder = _create_transformer_decoder()

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


def _load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    """Load model checkpoint from file."""
    # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    sam3_image_ckpt = {k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k}
    model.load_state_dict(sam3_image_ckpt, strict=False)


def _setup_device_and_mode(
    model: Sam3Image,
    device: str,
    eval_mode: bool,
) -> Sam3Image:
    """Setup model device, evaluation mode, and dtype."""
    if device == "cuda":
        model = model.cuda()
    elif device == "xpu":
        model = model.to("xpu")
    if eval_mode:
        model.eval()
    return model


def build_sam3_image_model(
    bpe_path: Path | None = None,
    device: str = "cuda",
    checkpoint_path: Path | None = None,
    load_from_hf: bool = True,
    enable_segmentation: bool = True,
    enable_inst_interactivity: bool = False,
    compile: bool = False,
) -> Sam3Image:
    """Build SAM3 image model

    Args:
        bpe_path: Path to the BPE tokenizer vocabulary
        device: Device to load the model on ('cuda' or 'cpu')
        checkpoint_path: Optional path to model checkpoint
        load_from_hf: Whether to load the checkpoint from Hugging Face Hub
        enable_segmentation: Whether to enable segmentation head
        enable_inst_interactivity: Whether to enable instance interactivity (SAM 1 task)
        compile: To enable compilation, set to True

    Returns:
        A SAM3 image model

    Raises:
        FileNotFoundError: If the provided BPE path does not exist
    """
    if bpe_path is None:
        bpe_path = Path(__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    if not bpe_path.exists():
        msg = f"BPE path {bpe_path} does not exist"
        raise FileNotFoundError(msg)

    # Create visual components
    compile_mode = "default" if compile else None
    vision_encoder = _create_vision_backbone(
        compile_mode=compile_mode,
        enable_inst_interactivity=enable_inst_interactivity,
        device=device,
    )

    # Create text components
    text_encoder = _create_text_encoder(bpe_path)

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
    # Create the SAM3 model
    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
    )
    if load_from_hf and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    # Load checkpoint if provided
    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path)

    # Setup device and mode
    return _setup_device_and_mode(model, device, eval_mode=True)


def download_ckpt_from_hf() -> str:
    """Download SAM3 checkpoint from Hugging Face Hub."""
    sam3_model_id = "facebook/sam3"
    sam3_ckpt_name = "sam3.pt"
    sam3_cfg_name = "config.json"
    _ = hf_hub_download(repo_id=sam3_model_id, filename=sam3_cfg_name)
    return hf_hub_download(repo_id=sam3_model_id, filename=sam3_ckpt_name)
