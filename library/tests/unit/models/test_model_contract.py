# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Contract-level guarantees shared by every model.

These pin behaviour the documentation promises, independent of any one model:
model cards identify their own model, and the numpy->torch entry point accepts
every input form the public signatures advertise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from instantlearn.data.base.batch import Batch
from instantlearn.data.base.sample import Category, Sample
from instantlearn.models import (
    SAM3,
    EfficientSAM3,
    GroundedSAM,
    Matcher,
    MatcherOpenVINO,
    PerDino,
    PerDinoOpenVINO,
    SoftMatcher,
    SoftMatcherOpenVINO,
)
from instantlearn.models.torch_adapter import samples_to_tensors

_IMAGE_PATH = Path(__file__).parents[3] / "examples" / "assets" / "coco" / "000000286874.jpg"

# Every model that ships a card, paired with the family it should report.
_MODELS_AND_FAMILIES = [
    (SAM3, "sam3"),
    (EfficientSAM3, "efficient_sam3"),
    (Matcher, "matcher"),
    (PerDino, "per_dino"),
    (SoftMatcher, "soft_matcher"),
    (GroundedSAM, "grounded_sam"),
]

# OpenVINO siblings deliberately share their torch counterpart's card: a card
# describes what a model can do, not which runtime it happens to be using.
_OPENVINO_SIBLINGS = [
    (MatcherOpenVINO, Matcher),
    (PerDinoOpenVINO, PerDino),
    (SoftMatcherOpenVINO, SoftMatcher),
]


class TestModelCards:
    """Cards must identify the model they belong to."""

    @pytest.mark.parametrize(("model", "family"), _MODELS_AND_FAMILIES)
    def test_card_reports_its_own_family(self, model: type, family: str) -> None:
        """Each model reports its own family, not an inherited one."""
        assert model.card().family == family

    @pytest.mark.parametrize(("model", "family"), _MODELS_AND_FAMILIES)
    def test_card_is_readable_without_instantiation(self, model: type, family: str) -> None:
        """``card()`` is a classmethod, so capabilities are free to inspect."""
        card = model.card()

        assert card.name
        assert card.description
        assert card.prompt_types
        assert card.shot_modes
        assert family  # parametrization guard

    def test_efficient_sam3_does_not_inherit_sam3s_card(self) -> None:
        """EfficientSAM3 subclasses SAM3 but is a distinct model.

        Without its own card it silently reported ``name="SAM3"``, so callers
        selecting a model by name saw the wrong one.
        """
        assert EfficientSAM3.card() != SAM3.card()
        assert EfficientSAM3.card().name == "EfficientSAM3"

    @pytest.mark.parametrize(("openvino_model", "torch_model"), _OPENVINO_SIBLINGS)
    def test_openvino_sibling_shares_the_torch_card(self, openvino_model: type, torch_model: type) -> None:
        """Siblings describe one model with two runtimes."""
        assert openvino_model.card() == torch_model.card()


class TestSamplesToTensorsAcceptsCollatable:
    """The torch entry point must accept everything ``Collatable`` allows.

    ``samples_to_tensors()`` normalizes any ``Collatable`` input, including
    image paths, even though the public ``predict()``/``fit()`` signatures are
    typed ``Sample | list[Sample] | Batch``. Paths used to fail with
    ``AttributeError: 'str' object has no attribute 'image'`` because the
    conversion only handled ``Sample`` and ``Batch``.
    """

    @staticmethod
    def _sample() -> Sample:
        return Sample(image=np.zeros((8, 8, 3), dtype=np.uint8), categories=[Category(0, "thing")])

    def test_accepts_single_sample(self) -> None:
        """A lone ``Sample`` becomes a one-element list."""
        assert len(samples_to_tensors(self._sample())) == 1

    def test_accepts_sample_list(self) -> None:
        """A list is converted element-wise, preserving length."""
        assert len(samples_to_tensors([self._sample(), self._sample()])) == 2

    def test_accepts_batch(self) -> None:
        """A ``Batch`` is unwrapped to its samples."""
        assert len(samples_to_tensors(Batch.collate([self._sample(), self._sample()]))) == 2

    def test_accepts_image_path_string(self) -> None:
        """A path string is loaded rather than iterated character by character."""
        tensor_samples = samples_to_tensors(str(_IMAGE_PATH))

        assert len(tensor_samples) == 1
        assert tensor_samples[0].image is not None

    def test_accepts_path_object(self) -> None:
        """``pathlib.Path`` works the same as a string path."""
        assert len(samples_to_tensors(_IMAGE_PATH)) == 1

    def test_accepts_list_of_paths(self) -> None:
        """Each path in a list becomes its own sample."""
        tensor_samples = samples_to_tensors([str(_IMAGE_PATH), str(_IMAGE_PATH)])

        assert len(tensor_samples) == 2
        assert all(sample.image is not None for sample in tensor_samples)

    def test_images_are_chw_tensors(self) -> None:
        """Samples carry HWC numpy; the torch boundary transposes to CHW."""
        tensor_sample = samples_to_tensors(self._sample())[0]

        assert tensor_sample.image.shape == (3, 8, 8)
