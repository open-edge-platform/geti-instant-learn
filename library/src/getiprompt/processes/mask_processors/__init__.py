# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mask processors."""

from getiprompt.processes.mask_processors.mask_processor_base import MaskProcessor
from getiprompt.processes.mask_processors.mask_to_polygon import MasksToPolygons

__all__ = ["MaskProcessor", "MasksToPolygons"]
