# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Encoders."""

from .base_image_encoder import load_image_encoder
from .openvino_image_encoder import OpenVINOImageEncoder
from .pytorch_image_encoder import AVAILABLE_IMAGE_ENCODERS, PyTorchImageEncoder
from .timm import AVAILABLE_IMAGE_ENCODERS as TIMM_AVAILABLE_IMAGE_ENCODERS


__all__ = [
    "AVAILABLE_IMAGE_ENCODERS",
    "TIMM_AVAILABLE_IMAGE_ENCODERS",
    "OpenVINOImageEncoder",
    "PyTorchImageEncoder",
    "TimmImageEncoder",
    "load_image_encoder",
]
