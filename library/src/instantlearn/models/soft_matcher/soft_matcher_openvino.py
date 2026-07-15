# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SoftMatcher OpenVINO inference model.

``SoftMatcherOpenVINO`` runs the baked SoftMatcher IR (``model.xml``) produced by
:meth:`~instantlearn.models.soft_matcher.soft_matcher.SoftMatcher.to_openvino`
(inherited from ``Matcher``). The baked graph has the identical
``target_image -> (masks, scores, labels)`` IO contract as Matcher, so this class
is a trivial subclass of :class:`~instantlearn.models.matcher.matcher_openvino.MatcherOpenVINO`
that only overrides the model card.

Because the references are baked in, ``fit()`` is **not** supported — call
``SoftMatcher.fit(...)`` before ``SoftMatcher.to_openvino(...)`` to choose the
references, then load the resulting directory with ``SoftMatcherOpenVINO``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from instantlearn.models.matcher.matcher_openvino import MatcherOpenVINO

from ._card import _SOFT_MATCHER_CARD

if TYPE_CHECKING:
    from instantlearn.models.model_card import ModelCard


class SoftMatcherOpenVINO(MatcherOpenVINO):
    """SoftMatcher model running the baked OpenVINO IR for inference.

    Inherits the entire loader/inference implementation from
    :class:`~instantlearn.models.matcher.matcher_openvino.MatcherOpenVINO`
    (same ``model.xml`` IR layout and IO names); only the capability card
    differs.

    Examples:
        >>> from instantlearn.models.soft_matcher import SoftMatcher, SoftMatcherOpenVINO
        >>> from instantlearn.data.base.sample import Sample

        >>> # 1. Fit references and export the baked IR with a torch SoftMatcher.
        >>> soft_matcher = SoftMatcher(device="cpu")
        >>> soft_matcher.fit(Sample(image_path="ref.jpg", mask_paths=["mask.png"]))
        >>> ir_dir = soft_matcher.to_openvino("./softmatcher-ov")

        >>> # 2. Load and run the baked IR (no fit needed).
        >>> ov_model = SoftMatcherOpenVINO(model_dir=ir_dir, device="CPU")
        >>> predictions = ov_model.predict(Sample(image_path="target.jpg"))
    """

    @classmethod
    def card(cls) -> ModelCard:
        """Return the static capability descriptor for SoftMatcher."""
        return _SOFT_MATCHER_CARD
