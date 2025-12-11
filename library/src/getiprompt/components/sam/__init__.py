# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM predictor implementations for different backends."""

from .base import SAMPredictor, load_sam_model
from .openvino import OpenVINOSAMPredictor
from .pytorch import PyTorchSAMPredictor

__all__ = [
    "OpenVINOSAMPredictor",
    "PyTorchSAMPredictor",
    "SAMPredictor",
    "load_sam_model",
]
