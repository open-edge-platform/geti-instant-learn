# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Intermediate base class for all PyTorch-backed instantlearn models."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from torch import nn

from instantlearn.data.base.prediction import Prediction
from instantlearn.models.base import Model
from instantlearn.models.torch_adapter import sample_to_tensors
from instantlearn.utils.constants import Backend

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from instantlearn.data.base.batch import Batch
    from instantlearn.data.base.sample import Sample
    from instantlearn.models.openvino_base import OpenVINOModel
    from instantlearn.models.torch_adapter import TensorSample


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
    backend-neutral ``Model`` contract. Provides device/precision tracking, a
    ``_prepare()`` helper that converts ``Sample`` inputs to ``TensorSample``
    via the torch adapter, a ``_to_prediction()`` / ``_build_prediction()``
    pair that is the single torch->numpy boundary, and an abstract
    ``to_openvino()`` stub.

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

    def _prepare(self, target: Sample | list[Sample] | Batch) -> list[TensorSample]:
        """Convert ``Sample`` or ``Batch`` inputs to a list of ``TensorSample``.

        Calls :func:`sample_to_tensors` on each sample. Handles single samples,
        lists, and ``Batch`` objects uniformly.

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
        return [sample_to_tensors(s, self.device) for s in target]

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
        applied by ``_build_prediction``. The resulting ``Prediction`` is a pure
        numpy, backend-neutral data container.

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

    @staticmethod
    def _build_prediction(
        masks: np.ndarray,
        scores: np.ndarray,
        label_ids: np.ndarray,
        categories: Sequence[str],
        boxes: np.ndarray | None = None,
        points: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> Prediction:
        """Assemble a normalized numpy ``Prediction`` from raw numpy arrays.

        Enforces the contract dtypes:

        - ``masks``: ``bool`` if already boolean, otherwise ``uint8``.
        - ``scores``: ``float32``.
        - ``label_ids``: ``int32``.
        - ``boxes`` / ``points``: ``float32`` when present.

        ``label_names`` is derived by indexing ``categories`` with each entry
        of ``label_ids``; IDs outside the range fall back to ``str(id)``.

        Args:
            masks: Instance masks of shape ``(N, H, W)``.
            scores: Per-instance confidence scores of shape ``(N,)``.
            label_ids: Per-instance integer category IDs of shape ``(N,)``.
            categories: Sequence mapping a label ID to its category name.
            boxes: Optional bounding boxes of shape ``(N, 4)`` in xyxy format.
            points: Optional point predictions of shape ``(N, K, 2)``.
            metadata: Optional free-form per-prediction metadata.

        Returns:
            A ``Prediction`` with all arrays cast to the contract dtypes.
        """
        masks = np.ascontiguousarray(masks)
        if masks.dtype != np.bool_:
            masks = masks.astype(np.uint8, copy=False)

        scores = np.ascontiguousarray(scores, dtype=np.float32)
        label_ids = np.ascontiguousarray(label_ids, dtype=np.int32)

        n_categories = len(categories)
        label_names = np.array(
            [categories[i] if 0 <= i < n_categories else str(i) for i in label_ids.tolist()],
            dtype=object,
        )

        if boxes is not None:
            boxes = np.ascontiguousarray(boxes, dtype=np.float32)
        if points is not None:
            points = np.ascontiguousarray(points, dtype=np.float32)

        return Prediction(
            masks=masks,
            scores=scores,
            label_ids=label_ids,
            label_names=label_names,
            boxes=boxes,
            points=points,
            metadata=metadata if metadata is not None else {},
        )
