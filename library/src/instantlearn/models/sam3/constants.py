# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Canonical names for the SAM3 OpenVINO sub-model split.

Kept in a dependency-free module so the export and quantization CLIs can import
them without pulling in torch or transformers.
"""

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
