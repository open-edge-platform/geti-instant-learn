# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Feature extractors."""

from .masked_feature_extractor import MaskedFeatureExtractor
from .roi_cropper import CropRegion, CropResult, ROICropper

__all__ = [
    "CropRegion",
    "CropResult",
    "MaskedFeatureExtractor",
    "ROICropper",
]
