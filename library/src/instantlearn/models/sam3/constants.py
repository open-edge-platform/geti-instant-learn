# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Canonical names and configuration types shared by the SAM3 backends.

Kept in a dependency-free module so the export and quantization CLIs, and the
OpenVINO backend running a pre-exported IR, can import them without pulling in
torch or transformers.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

SAM3_MODEL_ID = "facebook/sam3.1"
"""HuggingFace repo the reference SAM3 weights and tokenizer come from."""

# Canonical sub-model names for the OpenVINO export split.
VISION_ENCODER = "vision-encoder"
TEXT_ENCODER = "text-encoder"
GEOMETRY_ENCODER = "geometry-encoder"
GEOMETRY_ENCODER_EXEMPLAR = "geometry-encoder-exemplar"
PROMPT_DECODER = "prompt-decoder"

# Sub-models written by a complete SAM3 export, in conversion order.
MODEL_NAMES = (
    VISION_ENCODER,
    TEXT_ENCODER,
    GEOMETRY_ENCODER,
    GEOMETRY_ENCODER_EXEMPLAR,
    PROMPT_DECODER,
)


@dataclass
class CanvasConfig:
    """Configuration for SAM3 canvas mode.

    Canvas mode stitches reference and target images into a single canvas,
    runs detection, and extracts predictions from the target region.

    Args:
        split_ratio: Fraction of the canvas allocated to the reference strip.
            Lower values give more space to the target image. Must be in
            (0, 1). Default: 0.3.
        crop_padding: Padding factor around the reference bounding box when
            cropping. A factor of 2.0 means the crop region is 2x the bbox
            size. Must be positive. Default: 2.0.
        cache_text: Cache text embeddings across canvas forward passes to
            avoid redundant CLIP encoding. Default: True.
        share_vision: Vision sharing strategy for multi-category canvas mode.
            - ``"auto"``: Groups same-category refs together with gaps between
              categories (equivalent to ``"grouped"``).
            - ``"grouped"``: Same-category refs packed side-by-side, gaps only
              between category groups.
            - ``"spaced"``: Each ref in its own slot with gaps between all refs.
            - ``False``: Sequential per-category canvases (no sharing).

    Examples:
        Use defaults:

        >>> config = CanvasConfig()

        Tune split ratio for small reference objects:

        >>> config = CanvasConfig(split_ratio=0.25, crop_padding=3.0)
    """

    split_ratio: float = 0.3
    crop_padding: float = 2.0
    cache_text: bool = True
    share_vision: Literal["auto", "grouped", "spaced"] | bool = "auto"

    def __post_init__(self) -> None:
        """Validate canvas configuration values.

        Raises:
            ValueError: If a configuration value is outside its supported range.
        """
        if not 0 < self.split_ratio < 1:
            msg = f"split_ratio must be in (0, 1), got {self.split_ratio}"
            raise ValueError(msg)
        if self.crop_padding <= 0:
            msg = f"crop_padding must be positive, got {self.crop_padding}"
            raise ValueError(msg)
        if not isinstance(self.share_vision, bool) and self.share_vision not in {
            "auto",
            "grouped",
            "spaced",
        }:
            msg = f'share_vision must be a bool or one of {{"auto", "grouped", "spaced"}}, got {self.share_vision!r}'
            raise ValueError(msg)


class Sam3PromptMode(str, Enum):
    """Prompt mode for SAM3 inference.

    Attributes:
        CLASSIC: Original SAM3 behavior. Text/box prompts are provided per target
            image. Boxes are encoded against the target image's own features.
        VISUAL_EXEMPLAR: Cross-image visual query detection. Box prompts on a
            reference image are encoded during fit() and reused for all target
            images. Enables "draw box on image A → detect similar on images B, C, D".
        CANVAS: FSS-SAM3 unified canvas approach. Stitches reference and target
            images into a single canvas, runs CLASSIC mode with the reference bbox
            mapped to canvas coordinates. Best visual-only performance.
    """

    CLASSIC = "classic"
    VISUAL_EXEMPLAR = "visual_exemplar"
    CANVAS = "canvas"
