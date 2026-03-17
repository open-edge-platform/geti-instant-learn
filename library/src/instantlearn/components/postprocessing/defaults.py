# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Default post-processing pipeline factory."""

from .base import PostProcessorPipeline
from .filtering import ScoreFilter
from .nms import BoxIoMNMS, MaskIoMNMS


def default_postprocessor() -> PostProcessorPipeline:
    """Return the standard default post-processing pipeline.

    The default pipeline first discards invalid masks, then removes
    nested/overlapping predictions:

    1. :class:`ScoreFilter` — removes masks with score ≤ 0 (zombie
       masks produced by the decoder's ``confidence_threshold``
       zeroing, important for ONNX/OpenVINO exports).
    2. :class:`MaskIoMNMS` — suppresses masks where ≥50% of the
       smaller mask is contained in a larger one.
    3. :class:`BoxIoMNMS` — same logic on bounding boxes for any
       remaining box-level overlaps.

    Returns:
        A :class:`PostProcessorPipeline` ready to attach to any model.

    Examples:
        >>> from instantlearn.components.postprocessing import default_postprocessor
        >>> pp = default_postprocessor()
        >>> len(pp)
        3
    """
    return PostProcessorPipeline([
        ScoreFilter(min_score=0.0),
        MaskIoMNMS(iom_threshold=0.5),
        BoxIoMNMS(iom_threshold=0.5),
    ])
