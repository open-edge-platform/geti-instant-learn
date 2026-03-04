# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Default post-processing pipeline factory."""

from .base import PostProcessorPipeline
from .nms import BoxIoMNMS, MaskIoMNMS


def default_postprocessor() -> PostProcessorPipeline:
    """Return the standard default post-processing pipeline.

    The default pipeline removes nested/overlapping predictions:

    1. :class:`MaskIoMNMS` — suppresses masks where ≥50% of the
       smaller mask is contained in a larger one.
    2. :class:`BoxIoMNMS` — same logic on bounding boxes for any
       remaining box-level overlaps.

    Returns:
        A :class:`PostProcessorPipeline` ready to attach to any model.

    Examples:
        >>> from instantlearn.components.postprocessing import default_postprocessor
        >>> pp = default_postprocessor()
        >>> len(pp)
        2
    """
    return PostProcessorPipeline([
        MaskIoMNMS(iom_threshold=0.5),
        BoxIoMNMS(iom_threshold=0.5),
    ])
