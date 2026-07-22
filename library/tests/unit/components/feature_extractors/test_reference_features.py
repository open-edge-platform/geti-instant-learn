# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ReferenceFeatures binary-mask guard."""

import pytest
import torch

from instantlearn.components.feature_extractors.reference_features import ReferenceFeatures


def _make(flatten_ref_masks: torch.Tensor) -> ReferenceFeatures:
    """Build a minimal ReferenceFeatures with the given flatten_ref_masks."""
    c, n, d = 1, flatten_ref_masks.shape[-1], 4
    return ReferenceFeatures(
        ref_embeddings=torch.zeros(c, n, d),
        masked_ref_embeddings=torch.zeros(c, d),
        flatten_ref_masks=flatten_ref_masks,
        category_ids=[0],
    )


class TestReferenceFeaturesGuard:
    """The guard must reject non-binary masks and accept binary ones."""

    def test_binary_int_mask_ok(self) -> None:
        """A strictly {0, 1} mask is accepted."""
        ref = _make(torch.tensor([[0.0, 1.0, 1.0, 0.0]]))
        assert ref.num_categories == 1

    def test_all_zero_mask_ok(self) -> None:
        """An all-zero mask is still binary and accepted."""
        ref = _make(torch.zeros(1, 4))
        assert ref.flatten_ref_masks.sum() == 0

    def test_scaled_mask_rejected(self) -> None:
        """A mask rescaled by 1/255 (the 'sky bug') is rejected."""
        with pytest.raises(ValueError, match="binary"):
            _make(torch.tensor([[0.0, 1.0 / 255.0, 1.0 / 255.0, 0.0]]))

    def test_fractional_mask_rejected(self) -> None:
        """Any fractional value raises."""
        with pytest.raises(ValueError, match="binary"):
            _make(torch.tensor([[0.0, 0.5, 1.0, 0.0]]))
