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
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np
    import torch

    from instantlearn.data.base.batch import Batch
    from instantlearn.data.base.prediction import Prediction
    from instantlearn.data.base.sample import Sample, TensorSample


class TorchModel(Model, nn.Module):
    """Intermediate base for all PyTorch-backed models.

    Provides device/precision tracking, a ``_prepare()`` helper that converts
    ``Sample`` inputs to ``TensorSample``, a ``_to_prediction()`` adapter that
    is the single torch->numpy boundary, and abstract stubs for ``export()``
    and ``to_openvino()``.

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
    def predict(self, target: Sample | list[Sample] | Batch) -> list[Prediction]:
        """Run inference on one or more target samples.

        Subclasses implement this with torch internals but must return
        numpy-based ``Prediction`` objects. Use ``self._prepare(target)``
        to convert inputs to ``TensorSample`` objects and
        ``self._to_prediction(...)`` to convert outputs back to numpy.

        Args:
            target: One or more target ``Sample`` objects, or a ``Batch``.

        Returns:
            A list of ``Prediction`` objects, one per input sample.
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
        """Export and immediately load the OpenVINO sibling model.

        Each concrete model implements its own conversion (graph tracing,
        dynamic axes, and submodel splitting vary per model).

        Returns:
            An ``OpenVINOModel`` instance ready for inference.
        """

    def _prepare(self, target: Sample | list[Sample] | Batch) -> list[TensorSample]:
        """Convert ``Sample`` or ``Batch`` inputs to a list of ``TensorSample``.

        Calls ``Sample.to_tensors(self.device)`` on each sample. Handles
        single samples, lists, and ``Batch`` objects uniformly.

        Args:
            target: One or more samples, or a ``Batch``.

        Returns:
            A list of ``TensorSample`` objects on ``self.device``.
        """
        from instantlearn.data.base.batch import Batch as BatchClass  # noqa: PLC0415
        from instantlearn.data.base.sample import Sample as SampleClass  # noqa: PLC0415

        if isinstance(target, SampleClass):
            target = [target]
        elif isinstance(target, BatchClass):
            target = target.samples
        return [s.to_tensors(self.device) for s in target]

    def _to_prediction(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        label_ids: torch.Tensor,
        categories: Sequence[str],
        boxes: torch.Tensor | None = None,
        points: torch.Tensor | None = None,
        metadata: dict | None = None,
    ) -> Prediction:
        """Convert torch model outputs to a numpy ``Prediction``.

        This is the single torch->numpy boundary for all PyTorch-backed models.
        Every tensor is moved to host memory via ``detach().cpu().numpy()``,
        then dtype normalization and ``label_ids -> label_names`` mapping are
        applied by ``Model._build_prediction``.
        OpenVINO models produce the same ``Prediction`` shape by calling
        ``_build_prediction`` directly on numpy arrays, so the app always
        receives an identical structure regardless of backend.

        Args:
            masks: Instance masks tensor of shape ``(N, H, W)``.
            scores: Confidence scores tensor of shape ``(N,)``.
            label_ids: Integer category IDs tensor of shape ``(N,)``.
            categories: Sequence mapping a label ID to its category name.
            boxes: Optional bounding boxes tensor of shape ``(N, 4)``.
            points: Optional point predictions tensor of shape ``(N, K, 2)``.
            metadata: Optional free-form per-prediction metadata.

        Returns:
            A numpy ``Prediction`` with contract dtypes enforced.
        """

        def _np(t: torch.Tensor | None) -> np.ndarray | None:
            return t.detach().cpu().numpy() if t is not None else None

        return self._build_prediction(
            masks=_np(masks),
            scores=_np(scores),
            label_ids=_np(label_ids),
            categories=categories,
            boxes=_np(boxes),
            points=_np(points),
            metadata=metadata,
        )
