# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Intermediate base class for all PyTorch-backed instantlearn models."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from torch import nn

from instantlearn.models.base import Model
from instantlearn.utils.constants import Backend

if TYPE_CHECKING:
    from pathlib import Path

    from instantlearn.data.base.prediction import Prediction
    from instantlearn.data.base.sample import Sample, TensorSample


class TorchModel(Model, nn.Module):
    """Intermediate base for all PyTorch-backed models.

    Adds torch-specific boilerplate: device/precision tracking, a ``_prepare``
    helper that converts :class:`~instantlearn.data.base.sample.Sample` inputs
    to :class:`~instantlearn.data.base.sample.TensorSample`, and abstract
    stubs for ``export`` / ``to_openvino``.

    Args:
        device: Torch device string, e.g. ``"cpu"`` or ``"cuda"``.
        precision: Weight precision string, e.g. ``"fp32"`` or ``"fp16"``.
        preprocessor: Optional numpy-based preprocessor applied before input.
        postprocessor: Optional post-processor applied after inference.
    """

    def __init__(
        self,
        device: str = "cpu",
        precision: str = "fp32",
        preprocessor: Any = None,  # noqa: ANN401
        postprocessor: Any = None,  # noqa: ANN401
    ) -> None:
        """Initialise with device, precision, and optional processors."""
        super().__init__()
        self.device = device
        self.precision = precision
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    @property
    def backend(self) -> Backend:
        """Always :attr:`~instantlearn.utils.constants.Backend.TORCH`."""
        return Backend.TORCH

    @abstractmethod
    def predict(self, target: Sample | list[Sample]) -> list[Prediction]:
        """Run inference.

        Subclasses implement this with torch internals but must return
        numpy-based :class:`~instantlearn.data.base.prediction.Prediction`
        objects.
        """

    @abstractmethod
    def export(self, path: Path) -> Path:
        """Export this model to OpenVINO IR.

        Args:
            path: Destination directory.

        Returns:
            Path to the exported ``.xml`` file.
        """

    @abstractmethod
    def to_openvino(self) -> Model:
        """Export and load the OpenVINO sibling model.

        Each concrete model implements its own conversion (graph tracing,
        dynamic axes, submodel splitting vary per model).

        Returns:
            An :class:`~instantlearn.models.openvino_base.OpenVINOModel`
            instance ready for inference.
        """

    # ------------------------------------------------------------------
    # Helpers

    def _prepare(self, target: Sample | list[Sample]) -> list[TensorSample]:
        """Convert Sample(s) to a list of TensorSample.

        Calls :meth:`~instantlearn.data.base.sample.Sample.to_tensors` on
        each sample using ``self.device``.

        Args:
            target: Single sample or list of samples.

        Returns:
            List of :class:`~instantlearn.data.base.sample.TensorSample`.
        """
        from instantlearn.data.base.sample import Sample as SampleClass  # noqa: PLC0415

        if isinstance(target, SampleClass):
            target = [target]
        return [s.to_tensors(self.device) for s in target]
