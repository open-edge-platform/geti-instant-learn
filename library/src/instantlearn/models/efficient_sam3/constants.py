# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Constants and configuration for EfficientSAM3 model variants."""

from enum import StrEnum

# HuggingFace repository for EfficientSAM3 checkpoints
HF_REPO_ID = "Simon7108528/EfficientSAM3"
HF_SUBFOLDER = "stage1_all_converted"


class BackboneType(StrEnum):
    """Supported image encoder backbone families."""

    EFFICIENTVIT = "efficientvit"
    REPVIT = "repvit"
    TINYVIT = "tinyvit"


class TextEncoderType(StrEnum):
    """Supported text encoder types."""

    MOBILECLIP_S1 = "mobileclip_s1"
    SAM3 = "sam3"


# Backbone configuration: maps (backbone_type, variant) -> (timm_model_name, output_channels)
BACKBONE_CONFIG: dict[tuple[str, str], tuple[str, int]] = {
    # EfficientViT (MIT Han Lab)
    (BackboneType.EFFICIENTVIT, "b0"): ("efficientvit_b0.r224_in1k", 128),
    (BackboneType.EFFICIENTVIT, "b1"): ("efficientvit_b1.r224_in1k", 256),
    (BackboneType.EFFICIENTVIT, "b2"): ("efficientvit_b2.r224_in1k", 384),
    # RepViT
    (BackboneType.REPVIT, "m0_9"): ("repvit_m0_9.dist_450e_in1k", 384),
    (BackboneType.REPVIT, "m1_1"): ("repvit_m1_1.dist_450e_in1k", 512),
    (BackboneType.REPVIT, "m2_3"): ("repvit_m2_3.dist_450e_in1k", 640),
    # TinyViT
    (BackboneType.TINYVIT, "5m"): ("tiny_vit_5m_224.dist_in22k_ft_in1k", 320),
    (BackboneType.TINYVIT, "11m"): ("tiny_vit_11m_224.dist_in22k_ft_in1k", 448),
    (BackboneType.TINYVIT, "21m"): ("tiny_vit_21m_224.dist_in22k_ft_in1k", 576),
}


def get_checkpoint_filename(
    backbone_type: str,
    variant: str,
    text_encoder: str = "mobileclip_s1",
) -> str:
    """Get the HuggingFace checkpoint filename for a given model configuration.

    Args:
        backbone_type: One of 'efficientvit', 'repvit', 'tinyvit'.
        variant: Model size variant (e.g. 'b2', 'm1_1', '11m').
        text_encoder: Text encoder type. Only 'mobileclip_s1' supported for
            combined checkpoints.

    Returns:
        Checkpoint filename on HuggingFace.

    Raises:
        ValueError: If the backbone/variant/text_encoder combination is unsupported.
    """
    if text_encoder == TextEncoderType.MOBILECLIP_S1:
        # Combined image + text encoder checkpoints
        name_map: dict[tuple[str, str], str] = {
            (BackboneType.EFFICIENTVIT, "b0"): "efficient_sam3_efficientvit-b0_mobileclip_s1.pth",
            (BackboneType.EFFICIENTVIT, "b1"): "efficient_sam3_efficientvit-b1_mobileclip_s1.pth",
            (BackboneType.EFFICIENTVIT, "b2"): "efficient_sam3_efficientvit-b2_mobileclip_s1.pth",
            (BackboneType.REPVIT, "m0_9"): "efficient_sam3_repvit-m0_9_mobileclip_s1.pth",
            (BackboneType.REPVIT, "m1_1"): "efficient_sam3_repvit-m1_1_mobileclip_s1.pth",
            (BackboneType.REPVIT, "m2_3"): "efficient_sam3_repvit-m2_3_mobileclip_s1.pth",
            (BackboneType.TINYVIT, "5m"): "efficient_sam3_tinyvit_5m_mobileclip_s1.pth",
            (BackboneType.TINYVIT, "11m"): "efficient_sam3_tinyvit_11m_mobileclip_s1.pth",
            (BackboneType.TINYVIT, "21m"): "efficient_sam3_tinyvit_21m_mobileclip_s1.pth",
        }
        key = (backbone_type, variant)
        if key not in name_map:
            msg = f"No checkpoint for backbone={backbone_type}, variant={variant}, text={text_encoder}"
            raise ValueError(msg)
        return name_map[key]

    msg = f"Text encoder type '{text_encoder}' not supported for combined checkpoints"
    raise ValueError(msg)


# MobileCLIP-S1 text encoder configuration
MOBILECLIP_S1_CONFIG: dict = {
    "vocab_size": 49408,
    "dim": 512,
    "n_transformer_layers": 12,
    "n_heads_per_layer": 8,
    "ffn_multiplier_per_layer": 4.0,
    "context_length": 77,
    "norm_layer": "layer_norm_fp32",
    "causal_masking": False,
    "model_name": "base",
    "embed_dropout": 0.0,
    "no_scale_embedding": False,
    "no_pos_embedding": False,
}

# Student text encoder context length (input sequence length)
# The EfficientSAM3 student text encoder was distilled with 32-token input
# sequences, matching the teacher's effective context length. The MobileCLIP
# positional embeddings are initialized for 77 positions and interpolated down
# to 32 at runtime. Using a different context length produces suboptimal text
# features and degrades detection quality.
STUDENT_CONTEXT_LENGTH = 32

# Student image encoder output config
IMAGE_ENCODER_EMBED_DIM = 1024  # Project backbone output to this dimension
IMAGE_ENCODER_EMBED_SIZE = 72  # Spatial size of encoded features (matches SAM3 ViT output)
IMAGE_ENCODER_IMAGE_SIZE = 1008  # Input image resolution

# TinyViT architecture configs per variant (from SAM-HQ registrations).
# SAM-HQ's timm factory functions ignore img_size, so we instantiate TinyViT
# directly and pass these configs along with the runtime image_size.
TINYVIT_CONFIGS: dict[str, dict] = {
    "5m": {
        "embed_dims": [64, 128, 160, 320],
        "depths": [2, 2, 6, 2],
        "num_heads": [2, 4, 5, 10],
        "window_sizes": [7, 7, 14, 7],
        "drop_path_rate": 0.0,
    },
    "11m": {
        "embed_dims": [64, 128, 256, 448],
        "depths": [2, 2, 6, 2],
        "num_heads": [2, 4, 8, 14],
        "window_sizes": [7, 7, 14, 7],
        "drop_path_rate": 0.1,
    },
    "21m": {
        "embed_dims": [96, 192, 384, 576],
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 18],
        "window_sizes": [7, 7, 14, 7],
        "drop_path_rate": 0.2,
    },
}
