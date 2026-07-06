# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Intermediate base class for all PyTorch-backed instantlearn models."""

from __future__ import annotations

from typing import Any

from torch import nn

from instantlearn.models.base import Model
from instantlearn.utils.constants import Backend


class TorchModel(nn.Module, Model):
    """Intermediate base for all PyTorch-backed models.

    Inherits ``nn.Module`` first so ``super().__init__()`` initializes the
    PyTorch internals (``_modules``, parameters, buffers) before the
    backend-neutral ``Model`` contract. Provides device/precision tracking and
    an abstract ``to_openvino()`` stub.

    Subclasses convert inputs and outputs through the torch adapter directly:
    :func:`~instantlearn.models.torch_adapter.samples_to_tensors` for inputs and
    :func:`~instantlearn.models.torch_adapter.tensors_to_prediction` (or
    :func:`~instantlearn.models.torch_adapter.arrays_to_prediction`) for the
    torch->numpy ``Prediction`` boundary.

    Attributes:
        device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
        precision: Weight precision string (e.g. ``"fp32"``, ``"fp16"``).
        preprocessor: Optional numpy-based preprocessor applied before inference.
        postprocessor: Optional post-processor applied after inference.
    """

    def __init__(
        self,
        device: str = "cpu",
        precision: str = "fp32",
        preprocessor: Any = None,  # noqa: ANN401
        postprocessor: Any = None,  # noqa: ANN401
    ) -> None:
        """Initialize with device, precision, and optional processors.

        Args:
            device: Torch device string, e.g. ``"cpu"`` or ``"cuda"``.
            precision: Weight precision, e.g. ``"fp32"`` or ``"fp16"``.
            preprocessor: Optional numpy-based preprocessor.
            postprocessor: Optional post-processor.
        """
        super().__init__()
        self.device = device
        self.precision = precision
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    @property
    def backend(self) -> Backend:
        """Always ``Backend.TORCH``."""
        return Backend.TORCH

