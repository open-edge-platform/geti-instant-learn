# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Composable mask post-processing pipeline for segmentation models.

This package provides chainable ``nn.Module`` post-processors that
transform segmentation predictions (masks, scores, labels). They can
be attached to any model via the ``postprocessor`` parameter and are
automatically included in ONNX export when traceable.

Overlap resolution:
    - :class:`MaskNMS` — mask-IoU based NMS
    - :class:`BoxNMS` — box-IoU based NMS (fastest)
    - :class:`MaskIoMNMS` — mask-IoM based NMS (best for nested objects)
    - :class:`BoxIoMNMS` — box-IoM based NMS (fast + handles nesting)
    - :class:`SoftNMS` — Gaussian score decay
    - :class:`PanopticArgmaxAssignment` — pixel-level argmax overlap resolution

Score filtering:
    - :class:`ScoreFilter` — remove zero/low-score masks

Mask cleaning:
    - :class:`MinimumAreaFilter` — area threshold filter
    - :class:`MorphologicalOpening` — remove small protrusions (ONNX-safe)
    - :class:`MorphologicalClosing` — fill small holes (ONNX-safe)
    - :class:`ConnectedComponentFilter` — remove small blobs (eager only)
    - :class:`HoleFilling` — fill enclosed holes (eager only)

Mask merging:
    - :class:`InstanceMerge` — merge spatially connected same-label masks (eager only)
    - :class:`MergePerClassMasks` — OR-merge masks per class (one mask per label)

Composition:
    - :class:`PostProcessorPipeline` — chain multiple processors
    - :func:`apply_postprocessing` — helper to apply to prediction dicts

Examples:
    >>> from instantlearn.components.postprocessing import (
    ...     MaskNMS,
    ...     MinimumAreaFilter,
    ...     MorphologicalOpening,
    ...     PanopticArgmaxAssignment,
    ...     PostProcessorPipeline,
    ... )
    >>> pipeline = PostProcessorPipeline([
    ...     MaskNMS(iou_threshold=0.5),
    ...     PanopticArgmaxAssignment(),
    ...     MinimumAreaFilter(min_area=64),
    ...     MorphologicalOpening(kernel_size=3),
    ... ])
"""

from .base import PostProcessor, PostProcessorPipeline, apply_postprocessing
from .connected_components import ConnectedComponentFilter, HoleFilling
from .filtering import MinimumAreaFilter, ScoreFilter
from .merge import InstanceMerge, MergePerClassMasks
from .morphology import MorphologicalClosing, MorphologicalOpening
from .nms import BoxIoMNMS, BoxNMS, MaskIoMNMS, MaskNMS, SoftNMS
from .overlap import PanopticArgmaxAssignment

__all__ = [
    "BoxIoMNMS",
    "BoxNMS",
    "ConnectedComponentFilter",
    "HoleFilling",
    "InstanceMerge",
    "MaskIoMNMS",
    "MaskNMS",
    "MergePerClassMasks",
    "MinimumAreaFilter",
    "MorphologicalClosing",
    "MorphologicalOpening",
    "PanopticArgmaxAssignment",
    "PostProcessor",
    "PostProcessorPipeline",
    "ScoreFilter",
    "SoftNMS",
    "apply_postprocessing",
]
