# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Backend-agnostic base class for all instantlearn models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from instantlearn.data.base.prediction import Prediction

if TYPE_CHECKING:
    from collections.abc import Sequence

    from instantlearn.data.base.batch import Batch
    from instantlearn.data.base.sample import Sample
    from instantlearn.models.model_card import ModelCard
    from instantlearn.utils.constants import Backend


class Model(ABC):
    """Abstract base for all instantlearn models.

    The contract is backend-agnostic: inputs and outputs use numpy-based ``Sample`` / ``Prediction`` objects
    so the API works without torch.

    Concrete implementations inherit from one of two intermediate bases:

    - ``TorchModel`` — PyTorch-backed.
    - ``OpenVINOModel`` — OpenVINO-backed.

    Example:
        >>> model = SAM3()
        >>> model.fit(Sample(image=ref_img, masks=ref_masks, categories=["cat"]))
        >>> preds = model.predict([Sample(image=img1), Sample(image=img2)])
    """

    @classmethod
    @abstractmethod
    def card(cls) -> ModelCard:
        """Return the static ``ModelCard`` describing this model's capabilities.

        OV siblings delegate to their torch counterpart::

            @classmethod
            def card(cls) -> ModelCard:
                return SAM3.card()
        """

    @property
    @abstractmethod
    def backend(self) -> Backend:
        """The backend this instance is currently running on."""

    @abstractmethod
    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Load reference prompts or visual exemplars.

        Calling ``fit()`` again replaces the previous state (idempotent).
        Models that require a ``fit()`` call before ``predict()`` raise ``ModelNotFittedError``
        if ``predict()`` is called first.

        Args:
            reference: One or more reference ``Sample`` objects, or a
                ``Batch`` of them.
        """

    @abstractmethod
    def predict(self, target: Sample | list[Sample] | Batch) -> list[Prediction]:
        """Run inference on one or more target samples.

        Args:
            target: One or more target ``Sample`` objects, or a ``Batch``.

        Returns:
            A list of ``Prediction`` objects, one per input sample.
        """

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
        """Assemble a normalized numpy ``Prediction`` from raw arrays.

        Shared by every backend so the output shape and dtypes are identical
        regardless of where inference ran. Enforces the contract dtypes:

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
            categories: Sequence mapping a label ID to its category name
                (``categories[label_id]``).
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
