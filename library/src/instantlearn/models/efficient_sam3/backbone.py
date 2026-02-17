# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Student vision backbone for EfficientSAM3.

Uses timm backbones (EfficientViT, RepViT, TinyViT) with a projection head
and FPN neck to produce multi-scale features compatible with SAM3's DETR pipeline.
"""

import logging

import timm
import torch
from torch import nn
from torch.nn import functional

from instantlearn.models.sam3.common import SinePositionEmbedding

from .constants import (
    BACKBONE_CONFIG,
    IMAGE_ENCODER_EMBED_DIM,
    IMAGE_ENCODER_EMBED_SIZE,
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
    ) -> None:
        """Initialize the student vision model.

        Args:
            backbone_type: Backbone family ('efficientvit', 'repvit', 'tinyvit').
            variant: Model size variant (e.g. 'b0', 'm1_1', '11m').
            fpn_hidden_size: FPN output dimension. Default: 256.

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

        # Create timm backbone with only the final stage output
        backbone_model = timm.create_model(
            timm_name,
            pretrained=False,
            features_only=True,
            out_indices=(-1,),
        )

        # Match original EfficientSAM3 hierarchy:
        # trunk (ImageStudentEncoder) → student_trunk (StudentBackboneTrunk) → model (timm)
        self.trunk = nn.Module()
        self.trunk.student_trunk = StudentBackboneTrunk(backbone_model)
        self.trunk.projector = ImageProjectionHead(in_channels=out_channels)

        # FPN components as direct attributes (matching Sam3DualViTDetNeck)
        self.position_encoding = SinePositionEmbedding(
            num_pos_feats=fpn_hidden_size // 2,
            normalize=True,
        )
        self.convs = nn.ModuleList()
        scale_factors = [4.0, 2.0, 1.0, 0.5]
        for scale in scale_factors:
            current = nn.Sequential()
            if scale == 4.0:
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
            elif scale == 2.0:
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
            elif scale == 1.0:
                out_dim = IMAGE_ENCODER_EMBED_DIM
            elif scale == 0.5:
                current.add_module("maxpool_2x2", nn.MaxPool2d(kernel_size=2, stride=2))
                out_dim = IMAGE_ENCODER_EMBED_DIM
            else:
                msg = f"Unsupported scale_factor={scale}"
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
        **kwargs: dict,  # noqa: ARG002
    ) -> dict[str, torch.Tensor | None]:
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
