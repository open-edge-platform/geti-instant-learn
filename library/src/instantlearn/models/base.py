# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Backend-agnostic base class for all instantlearn models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from instantlearn.data.base.batch import Batch
    from instantlearn.data.base.prediction import Prediction
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
