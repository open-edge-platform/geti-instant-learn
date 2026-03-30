# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Components."""

from .cosine_similarity import CosineSimilarity
from .encoders import ImageEncoder
from .linear_sum_assignment import linear_sum_assignment
from .negative_prompts import NegativeMaskToPoints
from .postprocessing import (
    BoxIoMNMS,
    BoxNMS,
    MaskIoMNMS,
    MaskNMS,
    MergePerClassMasks,
    MinimumAreaFilter,
    MorphologicalClosing,
    MorphologicalOpening,
    PostProcessor,
    PostProcessorPipeline,
    SoftNMS,
    apply_postprocessing,
    default_postprocessor,
)
from .sam.decoder import SamDecoder

__all__ = [
    "BoxIoMNMS",
    "BoxNMS",
    "CosineSimilarity",
    "ImageEncoder",
    "MaskIoMNMS",
    "MaskNMS",
    "MergePerClassMasks",
    "MinimumAreaFilter",
    "MorphologicalClosing",
    "MorphologicalOpening",
    "NegativeMaskToPoints",
    "PostProcessor",
    "PostProcessorPipeline",
    "SamDecoder",
    "SoftNMS",
    "apply_postprocessing",
    "default_postprocessor",
    "linear_sum_assignment",
]
