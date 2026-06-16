# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Backend-agnostic base class for all instantlearn models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from instantlearn.data.base.prediction import Prediction
    from instantlearn.data.base.sample import Sample
    from instantlearn.models.model_card import ModelCard
    from instantlearn.utils.constants import Backend


class Model(ABC):
    """Abstract base class for all models.

    The contract is backend-agnostic, concrete implementations inherit from one of the intermediate bases:

    * :class:`~instantlearn.models.torch_base.TorchModel` — PyTorch-backed.
    * :class:`~instantlearn.models.openvino_base.OpenVINOModel` — OV-backed.

    Usage::

        model = SAM3()
        model.fit(Sample(image=ref_img, masks=ref_masks, categories=["cat"]))
        preds = model.predict([Sample(image=img1), Sample(image=img2)])
    """

    @classmethod
    @abstractmethod
    def card(cls) -> ModelCard:
        """Return the static ModelCard for this model class.

        OV siblings should delegate to the torch sibling::

            @classmethod
            def card(cls) -> ModelCard:
                return SAM3.card()
        """

    @property
    @abstractmethod
    def backend(self) -> Backend:
        """The backend this instance is currently running on."""

    @abstractmethod
    def fit(self, reference: Sample | list[Sample]) -> None:
        """Load reference prompts / exemplars.

        Calling ``fit`` again replaces the previous state (idempotent).

        Args:
            reference: One or more reference
                :class:`~instantlearn.data.base.sample.Sample` objects.
        """

    @abstractmethod
    def predict(self, target: Sample | list[Sample]) -> list[Prediction]:
        """Run inference on one or more target samples.

        Args:
            target: One or more target
                :class:`~instantlearn.data.base.sample.Sample` objects.

        Returns:
            A list of :class:`~instantlearn.data.base.prediction.Prediction`
            objects, one per input sample.
        """
