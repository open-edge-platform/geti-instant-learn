# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Composable mask post-processing pipeline for segmentation models.

This package provides chainable ``nn.Module`` post-processors that
transform segmentation predictions (masks, scores, labels). They can
be attached to any model via the ``postprocessor`` parameter and are
automatically included in ONNX export.

All processors use pure PyTorch operations and are ONNX/OpenVINO exportable.

Overlap resolution:
    - :class:`MaskNMS` — mask-IoU based NMS
    - :class:`BoxNMS` — box-IoU based NMS (fastest)
    - :class:`MaskIoMNMS` — mask-IoM based NMS (best for nested objects)
    - :class:`BoxIoMNMS` — box-IoM based NMS (fast + handles nesting)
    - :class:`SoftNMS` — Gaussian score decay

Score filtering:
    - :class:`ScoreFilter` — remove zero/low-score masks

Mask cleaning:
    - :class:`MinimumAreaFilter` — area threshold filter
    - :class:`MorphologicalOpening` — remove small protrusions
    - :class:`MorphologicalClosing` — fill small holes

Mask merging:
    - :class:`MergePerClassMasks` — OR-merge masks per class (one mask per label)

Composition:
    - :class:`PostProcessorPipeline` — chain multiple processors
    - :func:`apply_postprocessing` — helper to apply to prediction dicts
    - :func:`default_postprocessor` — standard default pipeline

Examples:
    >>> from instantlearn.components.postprocessing import (
    ...     MaskIoMNMS,
    ...     MinimumAreaFilter,
    ...     MorphologicalOpening,
    ...     PostProcessorPipeline,
    ... )
    >>> pipeline = PostProcessorPipeline([
    ...     MaskIoMNMS(iom_threshold=0.5),
    ...     MinimumAreaFilter(min_area=64),
    ...     MorphologicalOpening(kernel_size=3),
    ... ])
"""

from .base import PostProcessor, PostProcessorPipeline, apply_postprocessing
from .defaults import default_postprocessor
from .filtering import MinimumAreaFilter, ScoreFilter
from .merge import MergePerClassMasks
from .morphology import MorphologicalClosing, MorphologicalOpening
from .nms import BoxIoMNMS, BoxNMS, MaskIoMNMS, MaskNMS, SoftNMS

__all__ = [
    "BoxIoMNMS",
    "BoxNMS",
    "MaskIoMNMS",
    "MaskNMS",
    "MergePerClassMasks",
    "MinimumAreaFilter",
    "MorphologicalClosing",
    "MorphologicalOpening",
    "PostProcessor",
    "PostProcessorPipeline",
    "ScoreFilter",
    "SoftNMS",
    "apply_postprocessing",
    "default_postprocessor",
]
