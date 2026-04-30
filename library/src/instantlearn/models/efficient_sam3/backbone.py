# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Student vision backbone for EfficientSAM3.

Uses timm backbones (EfficientViT, RepViT, TinyViT) with a projection head
and FPN neck to produce multi-scale features compatible with SAM3's DETR pipeline.
"""

import logging
import math
from typing import Any

import timm
import torch
from segment_anything_hq.modeling.tiny_vit_sam import TinyViT
from torch import nn
from torch.nn import functional

from instantlearn.models.sam3.common import SinePositionEmbedding

from .constants import (
    BACKBONE_CONFIG,
    IMAGE_ENCODER_EMBED_DIM,
    IMAGE_ENCODER_EMBED_SIZE,
    IMAGE_ENCODER_IMAGE_SIZE,
    TINYVIT_CONFIGS,
)

logger = logging.getLogger(__name__)


class ImageProjectionHead(nn.Module):
    """Projects student backbone features to SAM3-compatible embedding space.

    Applies Conv1x1 -> BatchNorm -> GELU -> Conv3x3 projection, then bilinear
    interpolation to produce fixed-size spatial features matching SAM3 ViT output.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = IMAGE_ENCODER_EMBED_DIM,
        embed_size: int = IMAGE_ENCODER_EMBED_SIZE,
    ) -> None:
        """Initialize the projection head.

        Args:
            in_channels: Number of input channels from the backbone.
            embed_dim: Output embedding dimension. Default: 1024.
            embed_size: Target spatial size (height=width). Default: 72.
        """
        super().__init__()
        self.embed_size = embed_size
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project and resize features.

        Args:
            features: Backbone output [B, C_in, H, W].

        Returns:
            Projected features [B, embed_dim, embed_size, embed_size].
        """
        features = self.head(features)
        if features.shape[-2] != self.embed_size or features.shape[-1] != self.embed_size:
            features = functional.interpolate(
                features,
                size=(self.embed_size, self.embed_size),
                mode="bilinear",
                align_corners=False,
            )
        return features


class StudentBackboneTrunk(nn.Module):
    """Wrapper for timm backbone matching EfficientSAM3 checkpoint naming.

    Uses attribute name ``model`` for the timm backbone, matching the original
    EfficientSAM3 StudentBackboneTrunk for checkpoint compatibility.
    """

    def __init__(self, backbone_model: nn.Module) -> None:
        """Initialize with a timm backbone model.

        Args:
            backbone_model: A timm model instance.
        """
        super().__init__()
        self.model = backbone_model

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract features from backbone.

        Args:
            x: Input images [B, 3, H, W].

        Returns:
            List of feature maps from backbone stages.
        """
        return self.model(x)


class TinyViTBackboneTrunk(nn.Module):
    """Wrapper for SAM-HQ TinyViT with manual feature extraction.

    SAM-HQ's timm registration ignores ``img_size`` and
    ``forward_features()`` hardcodes spatial dimensions (64*64), so we
    cannot use ``timm.create_model`` or the default forward path.
    This wrapper creates TinyViT directly with the correct ``img_size``
    and manually runs ``patch_embed`` + all stages to extract features.

    Uses attribute name ``model`` for checkpoint key compatibility with
    ``StudentBackboneTrunk``.
    """

    def __init__(self, variant: str, image_size: int) -> None:
        """Initialize with a SAM-HQ TinyViT model.

        Args:
            variant: TinyViT variant ('5m', '11m', '21m').
            image_size: Input image resolution (e.g. 1008).

        Raises:
            ValueError: If variant is not supported.
        """
        super().__init__()
        if variant not in TINYVIT_CONFIGS:
            msg = f"Unsupported TinyViT variant: {variant}. Available: {list(TINYVIT_CONFIGS.keys())}"
            raise ValueError(msg)

        self.model = TinyViT(img_size=image_size, **TINYVIT_CONFIGS[variant])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract last-stage features from TinyViT.

        Runs patch_embed + all layers manually, then reshapes the sequence
        output to [B, C, H, W] for compatibility with the projection head.

        Args:
            x: Input images [B, 3, H, W].

        Returns:
            Single-element list containing [B, C, H, W] features.
        """
        x = self.model.patch_embed(x)
        for layer in self.model.layers:
            x = layer(x)
        # x: [B, L, C] → [B, C, H, W]
        b, seq_len, c = x.shape
        h = w = math.isqrt(seq_len)
        return [x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()]


class StudentVisionModel(nn.Module):
    """Complete student vision model: timm backbone + projection head + FPN.

    Drop-in replacement for SAM3's VisionModel, producing the same output format.
    Attribute hierarchy mirrors the original Sam3DualViTDetNeck for checkpoint
    compatibility:
    - trunk.student_trunk.model.* → timm backbone weights
    - trunk.projector.* → projection head weights
    - position_encoding.* → FPN position encoding weights
    - convs.* → FPN conv layer weights
    """

    def __init__(
        self,
        backbone_type: str = "efficientvit",
        variant: str = "b2",
        fpn_hidden_size: int = 256,
        image_size: int = IMAGE_ENCODER_IMAGE_SIZE,
    ) -> None:
        """Initialize the student vision model.

        Args:
            backbone_type: Backbone family ('efficientvit', 'repvit', 'tinyvit').
            variant: Model size variant (e.g. 'b0', 'm1_1', '11m').
            fpn_hidden_size: FPN output dimension. Default: 256.
            image_size: Input image resolution. Passed to the timm backbone so
                that architectures with fixed-resolution bookkeeping (TinyViT)
                compute correct internal spatial dimensions. Default: 1008.

        Raises:
            ValueError: If backbone_type/variant combination is not supported.
        """
        super().__init__()
        key = (backbone_type, variant)
        if key not in BACKBONE_CONFIG:
            msg = f"Unsupported backbone: {backbone_type}/{variant}. Available: {list(BACKBONE_CONFIG.keys())}"
            raise ValueError(msg)

        timm_name, out_channels = BACKBONE_CONFIG[key]
        logger.info("Creating student backbone: %s (timm: %s, channels: %d)", key, timm_name, out_channels)

        # Match original EfficientSAM3 hierarchy:
        # trunk (ImageStudentEncoder) → student_trunk → model (backbone)
        self.trunk = nn.Module()

        if backbone_type == "tinyvit":
            # SAM-HQ's timm registration ignores img_size (always uses 224)
            # and forward_features() hardcodes spatial dims. Create TinyViT
            # directly with the correct img_size and extract features manually.
            self.trunk.student_trunk = TinyViTBackboneTrunk(
                variant=variant,
                image_size=image_size,
            )
        else:
            backbone_model = timm.create_model(
                timm_name,
                pretrained=False,
                features_only=True,
                out_indices=(-1,),
            )
            self.trunk.student_trunk = StudentBackboneTrunk(backbone_model)

        self.trunk.projector = ImageProjectionHead(in_channels=out_channels)

        # FPN components as direct attributes (matching Sam3DualViTDetNeck)
        self.position_encoding = SinePositionEmbedding(
            num_pos_feats=fpn_hidden_size // 2,
            normalize=True,
        )
        self.convs = nn.ModuleList()
        scale_4x, scale_2x, scale_1x, scale_half = 0, 1, 2, 3
        for scale_idx in range(4):
            current = nn.Sequential()
            if scale_idx == scale_4x:
                current.add_module(
                    "dconv_2x2_0",
                    nn.ConvTranspose2d(
                        IMAGE_ENCODER_EMBED_DIM,
                        IMAGE_ENCODER_EMBED_DIM // 2,
                        kernel_size=2,
                        stride=2,
                    ),
                )
                current.add_module("gelu", nn.GELU())
                current.add_module(
                    "dconv_2x2_1",
                    nn.ConvTranspose2d(
                        IMAGE_ENCODER_EMBED_DIM // 2,
                        IMAGE_ENCODER_EMBED_DIM // 4,
                        kernel_size=2,
                        stride=2,
                    ),
                )
                out_dim = IMAGE_ENCODER_EMBED_DIM // 4
            elif scale_idx == scale_2x:
                current.add_module(
                    "dconv_2x2",
                    nn.ConvTranspose2d(
                        IMAGE_ENCODER_EMBED_DIM,
                        IMAGE_ENCODER_EMBED_DIM // 2,
                        kernel_size=2,
                        stride=2,
                    ),
                )
                out_dim = IMAGE_ENCODER_EMBED_DIM // 2
            elif scale_idx == scale_1x:
                out_dim = IMAGE_ENCODER_EMBED_DIM
            elif scale_idx == scale_half:
                current.add_module("maxpool_2x2", nn.MaxPool2d(kernel_size=2, stride=2))
                out_dim = IMAGE_ENCODER_EMBED_DIM
            else:
                msg = f"Unsupported scale index={scale_idx}"
                raise ValueError(msg)

            current.add_module("conv_1x1", nn.Conv2d(out_dim, fpn_hidden_size, kernel_size=1))
            current.add_module(
                "conv_3x3",
                nn.Conv2d(fpn_hidden_size, fpn_hidden_size, kernel_size=3, padding=1),
            )
            self.convs.append(current)

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> dict[str, Any]:
        """Extract multi-scale vision features.

        Args:
            pixel_values: Input images [B, 3, H, W].
            **kwargs: Additional keyword arguments (unused, for API compatibility).

        Returns:
            dict matching SAM3 VisionModel output format with keys:
                fpn_hidden_states, fpn_position_encoding, last_hidden_state,
                hidden_states, attentions.

        Raises:
            ValueError: If pixel_values is None.
        """
        if pixel_values is None:
            msg = "pixel_values is required"
            raise ValueError(msg)

        # Extract backbone features
        features_list = self.trunk.student_trunk(pixel_values)
        features = features_list[-1]  # Last stage output [B, C, H, W]

        # Project to SAM3-compatible embedding space
        projected = self.trunk.projector(features)  # [B, 1024, 72, 72]

        # Generate multi-scale FPN features
        fpn_hidden_states: tuple[torch.Tensor, ...] = ()
        fpn_position_encoding: tuple[torch.Tensor, ...] = ()

        for conv in self.convs:
            fpn_output = conv(projected)
            fpn_hidden_states += (fpn_output,)
            pos_enc = self.position_encoding(
                fpn_output.shape,
                fpn_output.device,
                fpn_output.dtype,
            )
            fpn_position_encoding += (pos_enc,)

        # Create last_hidden_state in sequence format for compatibility
        last_hidden_state = projected.flatten(2).transpose(1, 2)  # [B, 72*72, 1024]

        return {
            "last_hidden_state": last_hidden_state,
            "fpn_hidden_states": fpn_hidden_states,
            "fpn_position_encoding": fpn_position_encoding,
            "hidden_states": None,
            "attentions": None,
        }
