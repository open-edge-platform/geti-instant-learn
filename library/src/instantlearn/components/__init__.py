# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Components."""

from .cosine_similarity import CosineSimilarity
from .encoders import ImageEncoder
from .linear_sum_assignment import linear_sum_assignment
from .postprocessing import (
    ConnectedComponentFilter,
    HoleFilling,
    MaskIoMNMS,
    MaskNMS,
    MergePerClassMasks,
    MinimumAreaFilter,
    MorphologicalClosing,
    MorphologicalOpening,
    PanopticArgmaxAssignment,
    PostProcessor,
    PostProcessorPipeline,
    SoftNMS,
    apply_postprocessing,
)
from .sam.decoder import SamDecoder

__all__ = [
    "ConnectedComponentFilter",
    "CosineSimilarity",
    "HoleFilling",
    "ImageEncoder",
    "MaskIoMNMS",
    "MaskNMS",
    "MergePerClassMasks",
    "MinimumAreaFilter",
    "MorphologicalClosing",
    "MorphologicalOpening",
    "PanopticArgmaxAssignment",
    "PostProcessor",
    "PostProcessorPipeline",
    "SamDecoder",
    "SoftNMS",
    "apply_postprocessing",
    "linear_sum_assignment",
]
