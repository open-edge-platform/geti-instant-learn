# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Intermediate base class for all PyTorch-backed instantlearn models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import nn

from instantlearn.models.base import Model
from instantlearn.utils.constants import Backend, CompressionMode


@dataclass
class ExportConfig:
    """Options controlling Torch -> OpenVINO conversion.

    The conversion itself lives in each model's dedicated export script (e.g.
    ``instantlearn.scripts.matcher.export_matcher``) and is invoked by the
    OpenVINO sibling's ``from_torch()`` classmethod — ``TorchModel`` no longer
    exposes a ``to_openvino()`` method.

    Attributes:
        compression: OpenVINO weight compression mode of the exported IR.
            Defaults to ``CompressionMode.INT8_SYM``.
        opset: ONNX opset version for the intermediate graph.
        dynamic_shapes: Export with dynamic batch/spatial dims vs. static.
        keep_intermediate: Keep the intermediate ``.onnx`` files after IR
            conversion (useful for debugging).
    """

    compression: CompressionMode = field(default=CompressionMode.INT8_SYM)
    opset: int = 17
    dynamic_shapes: bool = True
    keep_intermediate: bool = False


class TorchModel(nn.Module, Model):
    """Intermediate base for all PyTorch-backed models.

    Inherits ``nn.Module`` first so ``super().__init__()`` initializes the
    PyTorch internals (``_modules``, parameters, buffers) before the
    backend-neutral ``Model`` contract. Provides device/precision tracking.

    Torch -> OpenVINO conversion is *not* a method on ``TorchModel``. Each
    OpenVINO sibling (e.g. ``MatcherOpenVINO``) owns the conversion via a
    ``from_torch()`` classmethod that delegates to the model's export script.
    This keeps the torch base free of OpenVINO concerns.

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
