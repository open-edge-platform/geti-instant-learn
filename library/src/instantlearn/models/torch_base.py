# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Intermediate base class for all PyTorch-backed instantlearn models."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from torch import nn

from instantlearn.models.base import Model
from instantlearn.utils.constants import Backend

if TYPE_CHECKING:
    from pathlib import Path

    from instantlearn.models.openvino_base import OpenVINOModel


@dataclass
class ExportConfig:
    """Options controlling Torch -> OpenVINO conversion.

    Attributes:
        precision: Weight/activation precision of the exported IR.
        opset: ONNX opset version for the intermediate graph.
        dynamic_shapes: Export with dynamic batch/spatial dims vs. static.
        keep_intermediate: Keep the intermediate ``.onnx`` files after IR
            conversion (useful for debugging).
    """

    precision: Literal["fp32", "fp16", "int8", "int4"] = "fp32"
    opset: int = 17
    dynamic_shapes: bool = True
    keep_intermediate: bool = False


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

    @abstractmethod
    def to_openvino(self, export_path: Path | None = None, config: ExportConfig | None = None) -> OpenVINOModel:
        """Export this Torch model to OpenVINO IR and load the OV sibling.

        Each concrete model implements its own conversion (graph tracing,
        dynamic axes, and submodel splitting vary per model). OpenVINO-specific
        behaviour is controlled through ``config``; values not covered by
        ``ExportConfig`` are inherited from this model's configuration.

        Args:
            export_path: Destination directory for the IR. ``None`` writes to a
                temporary directory.
            config: Export options (precision, opset, dynamic shapes, ...).
                ``None`` uses :class:`ExportConfig` defaults.

        Returns:
            An ``OpenVINOModel`` instance ready for inference.
        """
